# 3DTCMPE and CINR - Medical Image Processing Tools

This repository contains two main components:
1. 3DTCMPE for texture encoding
2. Cycle INR (CINR) for shape encoding


## Installation and Setup

Both tools can be installed in separate conda environments:

### 3DTCMPE Installation
```bash
# Create and activate conda environment
conda create -n 3dtcmpe python=3.8
conda activate 3dtcmpe

# Clone repository and install dependencies
git clone https://github.com/RichealYoung/DeepSTD
cd DeepSTD
pip install -r requirements.txt
```

### CINR Installation
```bash
# Create and activate conda environment
conda create -n cinr python=3.8
conda activate cinr

# Clone repository and install dependencies
git clone https://github.com/RichealYoung/DeepSTD
cd DeepSTD
pip install -r requirements.txt
```

## Basic Usage

### 3DTCMPE Usage
The 3DTCMPE model is designed for compressing 3D volumetric data:

1. Prepare your data in the following structure:
```
dataset/
├── train/
│   ├── volume1.npy
│   └── volume2.npy
└── test/
    ├── volume1.npy
    └── volume2.npy
```

2. Train the model:
```bash
python train.py \
    --dataset /path/to/dataset \
    --epochs 100 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --lambda 0.01 \
```

### CINR Usage
CINR performs deformable image registration using cycle-consistent implicit neural representations:

1. Prepare your medical images in NIfTI format:
   - Source image: `source.nii.gz`
   - Target image: `target.nii.gz`

2. Run registration:
```bash
python cycle_inr.py
```
