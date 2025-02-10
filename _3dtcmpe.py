from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """
    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update a registered buffer based on the policy.

    Args:
        module (nn.Module): the module containing the buffer
        buffer_name (str): name of the buffer in the module
        state_dict_key (str): key in the state dict corresponding to the buffer
        state_dict (dict): state dict containing the buffer data
        policy (str): update policy ("resize_if_empty", "resize", "register")
        dtype (torch.dtype): dtype of the buffer (default: torch.int)
    """
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized in a state dict.

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return

    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x


def conv1x1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1x1 3D convolution"""
    return nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3x3 3D convolution with padding"""
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3x3(in_ch: int, out_ch: int, upscale_factor: int = 2) -> nn.Module:
    """3x3x3 sub-pixel convolution for up-sampling"""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch * upscale_factor**3, kernel_size=3, padding=1),
        nn.PixelShuffle3d(upscale_factor),
    )


class PixelShuffle3d(nn.Module):
    """3D version of PixelShuffle."""

    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        channels //= self.upscale_factor**3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        x = x.view(
            batch_size,
            channels,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width,
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, channels, out_depth, out_height, out_width)

        return x


class ResidualBlock3D(nn.Module):
    """Basic 3D residual block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class ResidualBlockWithStride3D(nn.Module):
    """3D Residual block with a stride on the first convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv3x3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(out_ch, out_ch)
        self.skip = conv1x1x1(in_ch, out_ch, stride=stride)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        out = out + identity
        return out


class ResidualBlockUpsample3D(nn.Module):
    """3D Residual block with up-sampling."""

    def __init__(self, in_ch: int, out_ch: int, upscale_factor: int = 2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3x3(in_ch, out_ch, upscale_factor)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv3x3x3(out_ch, out_ch)
        self.upscale_factor = upscale_factor
        if in_ch != out_ch:
            self.skip = conv1x1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)
        identity = F.interpolate(
            identity,
            scale_factor=self.upscale_factor,
            mode="trilinear",
            align_corners=False,
        )

        out = out + identity
        return out


class WMSA3D(nn.Module):
    """3D Window Multi-head Self-attention module"""

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type

        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # 3D relative position embedding
        self.relative_position_params = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1) * (2 * window_size - 1),
                self.n_heads,
            )
        )

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(
                2 * window_size - 1,
                2 * window_size - 1,
                2 * window_size - 1,
                self.n_heads,
            ).permute(3, 0, 1, 2)
        )

    def generate_mask(self, d, h, w, p, shift):
        """Generate 3D attention mask"""
        attn_mask = torch.zeros(
            d,
            h,
            w,
            p,
            p,
            p,
            p,
            p,
            p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask

        s = p - shift
        # 3D cyclic shift mask
        attn_mask[-1, :, :, :s, :, :, s:, :, :] = True
        attn_mask[-1, :, :, s:, :, :, :s, :, :] = True
        attn_mask[:, -1, :, :, :s, :, :, s:, :] = True
        attn_mask[:, -1, :, :, s:, :, :, :s, :] = True
        attn_mask[:, :, -1, :, :, :s, :, :, s:] = True
        attn_mask[:, :, -1, :, :, s:, :, :, :s] = True

        attn_mask = rearrange(
            attn_mask,
            "w1 w2 w3 p1 p2 p3 p4 p5 p6 -> 1 1 (w1 w2 w3) (p1 p2 p3) (p4 p5 p6)",
        )
        return attn_mask

    def forward(self, x):
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(
                    -(self.window_size // 2),
                    -(self.window_size // 2),
                    -(self.window_size // 2),
                ),
                dims=(1, 2, 3),
            )

        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) (w3 p3) c -> b w1 w2 w3 p1 p2 p3 c",
            p1=self.window_size,
            p2=self.window_size,
            p3=self.window_size,
        )

        d_windows = x.size(1)
        h_windows = x.size(2)
        w_windows = x.size(3)

        x = rearrange(x, "b w1 w2 w3 p1 p2 p3 c -> b (w1 w2 w3) (p1 p2 p3) c")

        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)

        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale

        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")

        if self.type != "W":
            attn_mask = self.generate_mask(
                d_windows,
                h_windows,
                w_windows,
                self.window_size,
                shift=self.window_size // 2,
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)

        output = rearrange(
            output,
            "b (w1 w2 w3) (p1 p2 p3) c -> b (w1 p1) (w2 p2) (w3 p3) c",
            w1=d_windows,
            w2=h_windows,
            w3=w_windows,
            p1=self.window_size,
            p2=self.window_size,
            p3=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(
                    self.window_size // 2,
                    self.window_size // 2,
                    self.window_size // 2,
                ),
                dims=(1, 2, 3),
            )

        return output

    def relative_embedding(self):
        """Generate 3D relative position embedding"""
        cord = torch.tensor(
            np.array(
                [
                    [i, j, k]
                    for i in range(self.window_size)
                    for j in range(self.window_size)
                    for k in range(self.window_size)
                ]
            )
        )

        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[
            :,
            relation[:, :, 0].long(),
            relation[:, :, 1].long(),
            relation[:, :, 2].long(),
        ]


class Block3D(nn.Module):
    def __init__(
        self, input_dim, output_dim, head_dim, window_size, drop_path, type="W"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ["W", "SW"]
        self.type = type

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA3D(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock3D(nn.Module):
    """3D version of ConvTransBlock"""

    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type="W"):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type

        self.trans_block = Block3D(
            self.trans_dim,
            self.trans_dim,
            self.head_dim,
            self.window_size,
            self.drop_path,
            self.type,
        )

        self.conv1_1 = nn.Conv3d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            1,
            1,
            0,
            bias=True,
        )

        self.conv1_2 = nn.Conv3d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            1,
            1,
            0,
            bias=True,
        )

        self.conv_block = ResidualBlock3D(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(
            self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1
        )

        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = rearrange(trans_x, "b c d h w -> b d h w c")
        trans_x = self.trans_block(trans_x)
        trans_x = rearrange(trans_x, "b d h w c -> b c d h w")

        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x


class SWAtten3D(nn.Module):
    """3D version of Shifted Window Attention"""

    def __init__(
        self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.inter_dim = inter_dim

        if inter_dim is not None:
            self.in_conv = conv1x1x1(input_dim, inter_dim)
            self.out_conv = conv1x1x1(inter_dim, output_dim)
            self.non_local_block = SwinBlock3D(
                inter_dim, inter_dim, head_dim, window_size, drop_path
            )
        else:
            self.non_local_block = SwinBlock3D(
                input_dim, input_dim, head_dim, window_size, drop_path
            )

        self.conv_a = conv1x1x1(self.inter_dim, self.inter_dim)
        self.conv_b = conv1x1x1(self.inter_dim, self.inter_dim)

    def forward(self, x):
        if hasattr(self, "in_conv"):
            x = self.in_conv(x)

        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)

        out = a * torch.sigmoid(b)
        out += identity

        if hasattr(self, "out_conv"):
            out = self.out_conv(out)

        return out


class SwinBlock3D(nn.Module):
    """3D version of Swin Transformer Block"""

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path):
        super().__init__()
        self.block_1 = Block3D(
            input_dim, output_dim, head_dim, window_size, drop_path, type="W"
        )
        self.block_2 = Block3D(
            input_dim, output_dim, head_dim, window_size, drop_path, type="SW"
        )
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (
            (x.size(-1) <= self.window_size)
            or (x.size(-2) <= self.window_size)
            or (x.size(-3) <= self.window_size)
        ):
            padding_depth = (self.window_size - x.size(-3)) // 2
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(
                x,
                (
                    padding_col,
                    padding_col + 1,
                    padding_row,
                    padding_row + 1,
                    padding_depth,
                    padding_depth + 1,
                ),
            )
            resize = True

        trans_x = rearrange(x, "b c d h w -> b d h w c")
        trans_x = self.block_1(trans_x)
        trans_x = self.block_2(trans_x)
        trans_x = rearrange(trans_x, "b d h w c -> b c d h w")

        if resize:
            trans_x = F.pad(
                trans_x,
                (
                    -padding_col,
                    -padding_col - 1,
                    -padding_row,
                    -padding_row - 1,
                    -padding_depth,
                    -padding_depth - 1,
                ),
            )

        return trans_x


class TCM3D(CompressionModel):
    """3D Transform Coding with Masked Attention"""

    def __init__(
        self,
        config=[2, 2, 2, 2, 2, 2],
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0,
        N=128,
        M=320,
        num_slices=5,
        max_support_slices=5,
        **kwargs,
    ):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.N = N
        self.M = M

        # Calculate drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        # Encoder pathway
        self.m_down1 = nn.Sequential(
            *[
                ConvTransBlock3D(
                    N,
                    N,
                    self.head_dim[0],
                    self.window_size,
                    dpr[i + begin],
                    "W" if not i % 2 else "SW",
                )
                for i in range(config[0])
            ],
            ResidualBlockWithStride3D(2 * N, 2 * N, stride=2),
        )

        self.m_down2 = nn.Sequential(
            *[
                ConvTransBlock3D(
                    N,
                    N,
                    self.head_dim[1],
                    self.window_size,
                    dpr[i + begin],
                    "W" if not i % 2 else "SW",
                )
                for i in range(config[1])
            ],
            ResidualBlockWithStride3D(2 * N, 2 * N, stride=2),
        )

        self.m_down3 = nn.Sequential(
            *[
                ConvTransBlock3D(
                    N,
                    N,
                    self.head_dim[2],
                    self.window_size,
                    dpr[i + begin],
                    "W" if not i % 2 else "SW",
                )
                for i in range(config[2])
            ],
            conv3x3x3(2 * N, M, stride=2),
        )

        # Decoder pathway
        self.m_up1 = nn.Sequential(
            *[
                ConvTransBlock3D(
                    N,
                    N,
                    self.head_dim[3],
                    self.window_size,
                    dpr[i + begin],
                    "W" if not i % 2 else "SW",
                )
                for i in range(config[3])
            ],
            ResidualBlockUpsample3D(2 * N, 2 * N, 2),
        )

        self.m_up2 = nn.Sequential(
            *[
                ConvTransBlock3D(
                    N,
                    N,
                    self.head_dim[4],
                    self.window_size,
                    dpr[i + begin],
                    "W" if not i % 2 else "SW",
                )
                for i in range(config[4])
            ],
            ResidualBlockUpsample3D(2 * N, 2 * N, 2),
        )

        self.m_up3 = nn.Sequential(
            *[
                ConvTransBlock3D(
                    N,
                    N,
                    self.head_dim[5],
                    self.window_size,
                    dpr[i + begin],
                    "W" if not i % 2 else "SW",
                )
                for i in range(config[5])
            ],
            subpel_conv3x3x3(2 * N, 3, 2),
        )

        # Main encoder-decoder paths
        self.g_a = nn.Sequential(
            ResidualBlockWithStride3D(3, 2 * N, 2),
            *self.m_down1,
            *self.m_down2,
            *self.m_down3,
        )

        self.g_s = nn.Sequential(
            ResidualBlockUpsample3D(M, 2 * N, 2), *self.m_up1, *self.m_up2, *self.m_up3
        )

        # Hyperprior networks
        self.h_a = nn.Sequential(
            ResidualBlockWithStride3D(M, 2 * N, 2),
            *[
                ConvTransBlock3D(N, N, 32, 4, 0, "W" if not i % 2 else "SW")
                for i in range(config[0])
            ],
            conv3x3x3(2 * N, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample3D(192, 2 * N, 2),
            *[
                ConvTransBlock3D(N, N, 32, 4, 0, "W" if not i % 2 else "SW")
                for i in range(config[3])
            ],
            subpel_conv3x3x3(2 * N, M, 2),
        )

        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample3D(192, 2 * N, 2),
            *[
                ConvTransBlock3D(N, N, 32, 4, 0, "W" if not i % 2 else "SW")
                for i in range(config[3])
            ],
            subpel_conv3x3x3(2 * N, M, 2),
        )

        # Attention modules for mean and scale
        self.atten_mean = nn.ModuleList(
            [
                SWAtten3D(
                    M + (M // num_slices) * min(i, 5),
                    M + (M // num_slices) * min(i, 5),
                    16,
                    self.window_size,
                    0,
                    inter_dim=128,
                )
                for i in range(num_slices)
            ]
        )

        self.atten_scale = nn.ModuleList(
            [
                SWAtten3D(
                    M + (M // num_slices) * min(i, 5),
                    M + (M // num_slices) * min(i, 5),
                    16,
                    self.window_size,
                    0,
                    inter_dim=128,
                )
                for i in range(num_slices)
            ]
        )

        # Transform modules
        self.cc_mean_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        M + (M // num_slices) * min(i, 5),
                        224,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv3d(224, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv3d(
                        128, (M // num_slices), kernel_size=3, stride=1, padding=1
                    ),
                )
                for i in range(num_slices)
            ]
        )

        self.cc_scale_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        M + (M // num_slices) * min(i, 5),
                        224,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv3d(224, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv3d(
                        128, (M // num_slices), kernel_size=3, stride=1, padding=1
                    ),
                )
                for i in range(num_slices)
            ]
        )

        self.lrp_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        M + (M // num_slices) * min(i + 1, 6),
                        224,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv3d(224, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv3d(
                        128, (M // num_slices), kernel_size=3, stride=1, padding=1
                    ),
                )
                for i in range(num_slices)
            ]
        )

        # Initialize entropy models
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def load_state_dict(self, state_dict):
        """Handles loading of state dict for the model."""
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N=N // 2, M=M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.
        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.
        Args:
            scale_table (torch.Tensor): table of scales (default: None)
            force (bool): overwrite previous values (default: False)
        Returns:
            updated (bool): True if one of the CDFs was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        """Forward pass of the model.
        Args:
            x (torch.Tensor): input tensor [B, C, D, H, W]
        Returns:
            dict: containing the following keys:
                'x_hat': the reconstructed input tensor
                'likelihoods': contains the likelihoods of 'y' and 'z'
                'para': contains the parameters used for compression
        """
        # Encoder
        y = self.g_a(x)
        y_shape = y.shape[2:]

        # Hyperprior encoder
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Get quantized latents
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        # Get latent parameters
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        # Process slices
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Mean estimation
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1], : y_shape[2]]
            mu_list.append(mu)

            # Scale estimation
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1], : y_shape[2]]
            scale_list.append(scale)

            # Entropy coding
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            # Quantization
            y_hat_slice = ste_round(y_slice - mu) + mu

            # Local Refinement
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        # Reconstruct
        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y},
        }

    def compress(self, x):
        """Compresses an input tensor.
        Args:
            x (torch.Tensor): input tensor [B, C, D, H, W]
        Returns:
            dict: containing the following keys:
                'strings': bytes for 'y' and 'z'
                'shape': shape of z
        """
        # Encoder
        y = self.g_a(x)
        y_shape = y.shape[2:]

        # Hyperprior encoder
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-3:])

        # Get latent parameters
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        # Process slices
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        # Prepare for arithmetic coding
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Mean estimation
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1], : y_shape[2]]

            # Scale estimation
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1], : y_shape[2]]

            # Quantization and coding
            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            # Local refinement
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        # Encode symbols
        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-3:]}

    def decompress(self, strings, shape):
        """Decompresses a tensor from bytes.
        Args:
            strings (list): list of strings ['y', 'z']
            shape (tuple): shape of latent z
        Returns:
            dict: containing reconstructed tensor 'x_hat'
        """
        # Decompress hyperprior
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        # Set output shape
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4, z_hat.shape[4] * 4]

        # Prepare for decoding
        y_string = strings[0][0]
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Decode slices
        for slice_index in range(self.num_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Mean estimation
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1], : y_shape[2]]

            # Scale estimation
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1], : y_shape[2]]

            # Decode and dequantize
            index = self.gaussian_conditional.build_indexes(scale)
            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1], y_shape[2])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            # Local refinement
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        # Reconstruct
        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
