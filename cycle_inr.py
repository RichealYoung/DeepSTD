import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from torch.optim import Adam
import os


class SIREN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=3, num_layers=3):
        super().__init__()
        self.net = []
        self.net.append(nn.Linear(input_dim, hidden_dim))
        self.net.append(SineLayer())
        for _ in range(num_layers - 2):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(SineLayer())
        self.net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def forward(self, x):
        coords = x.clone()
        displacement = self.net(coords)
        return coords + displacement

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.data.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))


class SineLayer(nn.Module):
    def __init__(self, omega=30):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


class CycleConsistentRegistration:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.forward_field = SIREN().to(device)
        self.backward_field = SIREN().to(device)

    def load_images(self, source_path, target_path):
        source_nii = nib.load(source_path)
        target_nii = nib.load(target_path)
        self.source = torch.from_numpy(source_nii.get_fdata()).float().to(self.device)
        self.target = torch.from_numpy(target_nii.get_fdata()).float().to(self.device)
        self.source = (self.source - self.source.min()) / (
            self.source.max() - self.source.min()
        )
        self.target = (self.target - self.target.min()) / (
            self.target.max() - self.target.min()
        )
        self.source_affine = source_nii.affine
        self.target_affine = target_nii.affine
        self.grid = self.generate_grid(self.source.shape)

    def generate_grid(self, shape):
        x = torch.linspace(-1, 1, shape[0])
        y = torch.linspace(-1, 1, shape[1])
        z = torch.linspace(-1, 1, shape[2])
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(self.device)
        return grid.reshape(-1, 3)

    def compute_ncc(self, I, J):
        I_mean = I.mean()
        J_mean = J.mean()
        I_std = I.std()
        J_std = J.std()
        ncc = ((I - I_mean) * (J - J_mean)).mean() / (I_std * J_std)
        return -ncc

    def compute_jacobian_determinant(self, field, coords, batch_size=1000):
        num_points = coords.shape[0]
        loss_sum = torch.tensor(0.0, device=self.device)
        num_batches = (num_points + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)
            coords_batch = coords[start_idx:end_idx].clone()
            coords_batch.requires_grad_(True)
            displacement = field(coords_batch) - coords_batch
            jacobian = []
            for d in range(3):
                grad = torch.autograd.grad(
                    displacement[:, d].sum(), coords_batch, create_graph=True
                )[0]
                jacobian.append(grad)
            jacobian = torch.stack(jacobian, dim=1)
            jacobian = jacobian + torch.eye(3, device=self.device).unsqueeze(0)
            det = torch.linalg.det(jacobian)
            loss = torch.min(det - 1, torch.ones_like(det) * 10) ** 2 / det
            loss_sum += loss.sum()
            del jacobian, det, loss
            torch.cuda.empty_cache()
        return loss_sum / num_points

    def register(self, num_iterations=2500, learning_rate=1e-4, batch_size=4096):
        optimizer = Adam(
            list(self.forward_field.parameters())
            + list(self.backward_field.parameters()),
            lr=learning_rate,
        )
        best_loss = float("inf")
        best_iter = 0

        for i in range(num_iterations):
            optimizer.zero_grad()
            num_total_points = self.grid.shape[0]
            indices = torch.randperm(num_total_points, device=self.device)[:batch_size]
            sampled_grid = self.grid[indices]
            forward_coords = self.forward_field(sampled_grid)
            backward_coords = self.backward_field(sampled_grid)
            source_val = self.sample_image(self.source, forward_coords)
            target_val = self.sample_image(self.target, backward_coords)
            loss_sim_forward = self.compute_ncc(
                source_val, self.sample_image(self.target, sampled_grid)
            )
            loss_sim_backward = self.compute_ncc(
                target_val, self.sample_image(self.source, sampled_grid)
            )
            loss_cycle_forward = torch.norm(
                self.backward_field(forward_coords) - sampled_grid, dim=1
            ).mean()
            loss_cycle_backward = torch.norm(
                self.forward_field(backward_coords) - sampled_grid, dim=1
            ).mean()
            loss_reg_forward = self.compute_jacobian_determinant(
                self.forward_field, sampled_grid
            )
            loss_reg_backward = self.compute_jacobian_determinant(
                self.backward_field, sampled_grid
            )

            with torch.no_grad():
                disp_forward = torch.norm(forward_coords - sampled_grid, dim=1).mean()
                disp_backward = torch.norm(backward_coords - sampled_grid, dim=1).mean()

            alpha = 0.05
            beta = 0.001
            loss_sim = loss_sim_forward + loss_sim_backward
            loss_reg = alpha * (loss_reg_forward + loss_reg_backward)
            loss_cycle = beta * (loss_cycle_forward + loss_cycle_backward)
            total_loss = loss_sim + loss_reg + loss_cycle

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_iter = i + 1

            total_loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f"\nIteration {i+1}/{num_iterations}")
                print(
                    f"Total Loss: {total_loss.item():.4f} (Best: {best_loss:.4f} at iter {best_iter})"
                )
                print(f"Similarity Loss: {loss_sim.item():.4f}")
                print(f"Regularization Loss: {loss_reg.item():.4f}")
                print(f"Cycle Loss: {loss_cycle.item():.4f}")
                print(
                    f"Mean Displacement: forward={disp_forward.item():.2f}mm, backward={disp_backward.item():.2f}mm"
                )
                print(
                    f"NCC: forward={-loss_sim_forward.item():.4f}, backward={-loss_sim_backward.item():.4f}"
                )
                print("-" * 80)

    def sample_image(self, image, coords):
        coords = (coords + 1) / 2
        coords = coords * torch.tensor(
            image.shape, device=self.device
        ).float().unsqueeze(0)
        return torch.nn.functional.grid_sample(
            image.unsqueeze(0).unsqueeze(0),
            coords.reshape(1, -1, 1, 1, 3),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze()

    def save_displacement_field(self, save_path, field):
        with torch.no_grad():
            coords = self.grid.reshape(-1, 3)
            displacement = field(coords) - coords
            displacement = displacement.reshape(self.source.shape + (3,))
            displacement = displacement.cpu().numpy()
        nib.save(nib.Nifti1Image(displacement, self.source_affine), save_path)


def main():
    registrator = CycleConsistentRegistration()
    source_path = "source.nii.gz"
    target_path = "target.nii.gz"
    registrator.load_images(source_path, target_path)
    registrator.register()
    registrator.save_displacement_field(
        "forward_field.nii.gz", registrator.forward_field
    )
    registrator.save_displacement_field(
        "backward_field.nii.gz", registrator.backward_field
    )


if __name__ == "__main__":
    main()
