# 🏙️ Semantic Segmentation in Urban Scenes  

## 📌 Introduction  

Semantic segmentation is a core computer vision task that assigns a **class label to each pixel** in an image. This task is particularly **challenging in urban scene understanding** due to complex structures, varying lighting conditions, and occlusions.  

Deep learning has significantly advanced segmentation performance, with **U-Net and its extensions** being widely used due to their efficient **encoder-decoder architecture**. This project explores and compares the performance of three **segmentation models**:  

- **U-Net** 🏗 – A strong baseline with a simple yet effective architecture.  
- **Nested U-Net (U-Net++)** 🔗 – Uses **dense skip connections** to improve feature propagation.  
- **Attention U-Net** 🎯 – Incorporates **attention mechanisms** to enhance segmentation, particularly for fine details and occlusions.  

---

## 🎯 Objectives  

- Compare **U-Net, U-Net++, and Attention U-Net** on an **urban street dataset**.  
- Train models using a combination of **Cross-Entropy, Dice, and IoU losses**.  
- Evaluate segmentation accuracy in terms of **handling occlusions, fine details, and overall pixel-wise classification performance**.  
- Analyze the trade-offs between **model complexity and segmentation quality**.  

Repository: wageeshawlw/semantic-segmentation-with-u-net-and-its-variants
Files analyzed: 5

Estimated tokens: 44.2k

Directory structure:
└── wageeshawlw-semantic-segmentation-with-u-net-and-its-variants/
    ├── README.md
    ├── archs.py
    ├── attentionunet.py
    ├── Binary_Segmentation.ipynb
    └── Multiclass_Segmentation.ipynb


================================================
FILE: README.md
================================================
# 🏙️ Semantic Segmentation in Urban Scenes  

## 📌 Introduction  

Semantic segmentation is a core computer vision task that assigns a **class label to each pixel** in an image. This task is particularly **challenging in urban scene understanding** due to complex structures, varying lighting conditions, and occlusions.  

Deep learning has significantly advanced segmentation performance, with **U-Net and its extensions** being widely used due to their efficient **encoder-decoder architecture**. This project explores and compares the performance of three **segmentation models**:  

- **U-Net** 🏗 – A strong baseline with a simple yet effective architecture.  
- **Nested U-Net (U-Net++)** 🔗 – Uses **dense skip connections** to improve feature propagation.  
- **Attention U-Net** 🎯 – Incorporates **attention mechanisms** to enhance segmentation, particularly for fine details and occlusions.  

---

## 🎯 Objectives  

- Compare **U-Net, U-Net++, and Attention U-Net** on an **urban street dataset**.  
- Train models using a combination of **Cross-Entropy, Dice, and IoU losses**.  
- Evaluate segmentation accuracy in terms of **handling occlusions, fine details, and overall pixel-wise classification performance**.  
- Analyze the trade-offs between **model complexity and segmentation quality**.  



================================================
FILE: archs.py
================================================
import torch
from torch import nn

__all__ = ['UNet', 'NestedUNet','AttentionUnetLite']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
        
class ConvBlock(nn.Module):
    """Simplified convolution block with a single Conv layer, BatchNorm, and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownScale(nn.Module):
    """Downscaling with MaxPool followed by a single ConvBlock."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))

class AttentionBlock(nn.Module):
    """Simplified Attention Gate."""
    def __init__(self, f_g, f_l, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv_g = nn.Conv2d(f_g, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_x = nn.Conv2d(f_l, out_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        psi = self.sigmoid(self.psi(F.relu(self.conv_g(g) + self.conv_x(x))))
        return x * psi

class UpScale(nn.Module):
    """Upscaling with a single ConvBlock and an optional Attention Gate."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ConvBlock(in_channels, out_channels)
        self.attention = AttentionBlock(f_g=in_channels // 2, f_l=out_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ensure spatial dimensions match
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Apply attention before concatenation
        x2 = self.attention(x1, x2)

        # Concatenate and process
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionUnetLite(nn.Module):
    """Lightweight Attention U-Net with reduced depth and complexity."""
    def __init__(self, n_channels, n_classes, start=32, bilinear=False):
        super(AttentionUnetLite, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBlock(n_channels, start)
        self.down1 = DownScale(start, 2*start)
        self.down2 = DownScale(2*start, 4*start)

        factor = 2 if bilinear else 1
        self.down3 = DownScale(4*start, 8*start // factor)

        self.up1 = UpScale(8*start, 4*start // factor, bilinear)
        self.up2 = UpScale(4*start, 2*start, bilinear)
        self.up3 = UpScale(2*start, start, bilinear)
        self.outc = nn.Conv2d(start, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits



================================================
FILE: attentionunet.py
================================================

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(self.psi(g1 + x1))
        return x * psi


class UNetAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=23, init_features=64, dropout_rate=0.3):
        super(UNetAttention, self).__init__()
        features = init_features
        
        # Encoder Blocks
        self.encoder1 = self._block(in_channels, features, name="enc1", dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2", dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3", dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4", dropout_rate=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = self._block(features * 8, features * 16, name="enc5", dropout_rate=dropout_rate)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 16, features * 32, name="bottleneck", dropout_rate=dropout_rate)

        # Decoder Blocks
        self.upconv5 = nn.ConvTranspose2d(features * 32, features * 16, kernel_size=2, stride=2)
        self.att5 = AttentionGate(F_g=features * 16, F_l=features * 16, F_int=features * 8)
        self.decoder5 = self._block(features * 32, features * 16, name="dec5", dropout_rate=dropout_rate)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.decoder4 = self._block(features * 16, features * 8, name="dec4", dropout_rate=dropout_rate)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = self._block(features * 8, features * 4, name="dec3", dropout_rate=dropout_rate)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = self._block(features * 4, features * 2, name="dec2", dropout_rate=dropout_rate)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1", dropout_rate=dropout_rate)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck)
        enc5 = self.att5(dec5, enc5)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        enc4 = self.att4(dec4, enc4)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.att3(dec3, enc3)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.att2(dec2, enc2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        #return torch.sigmoid(self.conv(dec1)) # Use with BCEWithLogitsLoss
        return self.conv(dec1)

    def _block(self, in_channels, features, name, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # Add dropout
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )



================================================
FILE: Binary_Segmentation.ipynb
================================================
# Jupyter notebook converted to Python script.

import os
import pandas as pd
import numpy as np
import kagglehub
import shutil
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from accelerate import Accelerator
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from IPython.display import clear_output
from tqdm.auto import tqdm
from time import time
from typing import Tuple
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Output:
#   /usr/local/lib/python3.11/dist-packages/albumentations/__init__.py:24: UserWarning: A new version of Albumentations is available: 2.0.2 (you have 1.4.20). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.

#     check_for_updates()


# Download latest version
ebhi_path = kagglehub.dataset_download("mahdiislam/colorectal-cancer-wsi")

print("Path to dataset files:", ebhi_path)
# Output:
#   Downloading from https://www.kaggle.com/api/v1/datasets/download/mahdiislam/colorectal-cancer-wsi?dataset_version_number=1...

#   100%|██████████| 264M/264M [00:15<00:00, 17.6MB/s]
#   Extracting files...

#   

#   Path to dataset files: /root/.cache/kagglehub/datasets/mahdiislam/colorectal-cancer-wsi/versions/1


shutil.copytree(ebhi_path, "/content/EBHI")
# Output:
#   '/content/EBHI'

# Directory containing the dataset
directory = '/content/EBHI/EBHI-SEG'

# Output directories
output_train_images = '/content/EBHI/train/images'
output_train_masks = '/content/EBHI/train/masks'
output_val_images = '/content/EBHI/val/images'
output_val_masks = '/content/EBHI/val/masks'

# Create directories if they don't exist
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_train_masks, exist_ok=True)
os.makedirs(output_val_images, exist_ok=True)
os.makedirs(output_val_masks, exist_ok=True)

# Create a DataFrame to store file paths
df = pd.DataFrame(columns=['image_files', 'mask_files'])
cancer_types = ['Adenocarcinoma', 'High-grade IN', 'Low-grade IN', 'Normal', 'Polyp', 'Serrated adenoma']

for cancer_type in cancer_types:
    image_dir = os.path.join(directory, cancer_type, 'image')
    mask_dir = os.path.join(directory, cancer_type, 'label')

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Directory not found for {cancer_type}. Skipping...")
        continue

    image_files = sorted(os.listdir(image_dir))

    for file in image_files:
        image_file = os.path.join(image_dir, file)
        mask_file = os.path.join(mask_dir, file.replace('.jpg', '_mask.jpg'))

        if os.path.isfile(mask_file):
            df = pd.concat([df, pd.DataFrame({'image_files': [image_file], 'mask_files': [mask_file]})], ignore_index=True)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# Function to copy files to their respective directories
def copy_files(df, image_dest, mask_dest):
    for _, row in df.iterrows():
        shutil.copy(row['image_files'], os.path.join(image_dest, os.path.basename(row['image_files'])))
        shutil.copy(row['mask_files'], os.path.join(mask_dest, os.path.basename(row['mask_files'])))

# Copy training and validation files
copy_files(train_df, output_train_images, output_train_masks)
copy_files(val_df, output_val_images, output_val_masks)

# Function to count files in a directory
def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# Print the number of files in each folder
print(f"Train Images: {count_files(output_train_images)}")
print(f"Train Masks: {count_files(output_train_masks)}")
print(f"Validation Images: {count_files(output_val_images)}")
print(f"Validation Masks: {count_files(output_val_masks)}")


# Output:
#   Train Images: 1535

#   Train Masks: 1535

#   Validation Images: 663

#   Validation Masks: 663


class EbhiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask

class CONFIG:

    USE_MIXED_PRECISION = "fp16"
    DOWNSCALE = 2
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    EXTRA_LOSS_EPS = 1e-6
    BATCH_SIZE = 8
    SINGLE_NETWORK_TRAINING_EPOCHS = 20
    CE_VS_DICE_EVAL_EPOCHS = 15
    DELTA_BETA = 0.2

cfg = CONFIG()

preprocess = A.Compose([
    A.Normalize(mean=cfg.MEAN, std=cfg.STD, max_pixel_value=1.0),
    ToTensorV2(),
])


ebhi_train_ds = EbhiDataset(output_train_images, output_train_masks, transform=preprocess)
ebhi_val_ds = EbhiDataset(output_val_images, output_val_masks, transform=preprocess)
ebhi_train_dataloader = DataLoader(ebhi_train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
ebhi_val_dataloader = DataLoader(ebhi_val_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)


if cfg.USE_MIXED_PRECISION is not None:
    accelerator = Accelerator(mixed_precision=cfg.USE_MIXED_PRECISION)
else:
    accelerator = Accelerator()

import matplotlib.pyplot as plt
import cv2
import random
import os

def display_samples(image_dir, mask_dir):
    image_files = os.listdir(image_dir)

    # Select 4 random images
    random_images = random.sample(image_files, 4)

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # Reduce height to avoid excessive spacing

    for i, image_name in enumerate(random_images):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)  # Assuming mask names match image names

        # Read images
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale

        # Display image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")

        # Display mask
        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis("off")

    # Adjust spacing to minimize blank space
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()

# Call function to display samples
display_samples(output_train_images, output_train_masks)

# Output:
#   <Figure size 1200x600 with 8 Axes>

def decode_image(img: torch.Tensor) -> torch.Tensor:

    if img.shape[-1] == 3 and img.dim() == 3:
        img = img.permute(2, 0, 1)

    std = torch.tensor(cfg.STD, dtype=img.dtype, device=img.device).view(3, 1, 1)
    mean = torch.tensor(cfg.MEAN, dtype=img.dtype, device=img.device).view(3, 1, 1)

    img = img * std + mean

    if img.shape[0] == 3 and img.dim() == 3:
        img = img.permute(1, 2, 0)

    return torch.clamp(img, 0, 1)


category_cmap = 'grey'
eval_batch_data = next(iter(ebhi_val_dataloader))

def dice_coeff(inp: Tensor, tgt: Tensor, eps=1e-6):
    inter = 2 * (inp * tgt).sum()
    union = inp.sum() + tgt.sum()
    dice = (inter + eps) / (union + eps)
    return dice

def dice_loss(inp: Tensor, tgt: Tensor):
    return 1 - dice_coeff(inp, tgt)

def IoU_coeff(inp: Tensor, tgt: Tensor, eps=1e-6):
    inter = (inp * tgt).sum()
    union = inp.sum() + tgt.sum() - inter
    return (inter + eps) / (union + eps)

def IoU_loss(inp: Tensor, tgt: Tensor):
    return 1 - IoU_coeff(inp, tgt)


def evaluate_model(model, val_dataloader, epoch, epochs, criterion, with_dice=True, with_iou=True):
    val_loss = 0
    val_dice = 0
    val_iou = 0
    examples_so_far = 0

    model.eval()
    with tqdm(val_dataloader, desc=f"Epoch {epoch}/{epochs} - Validation Loss: 0") as pbar:
        for batch in val_dataloader:
            images, true_masks = batch[0].to(device), batch[1].to(device).float()
            true_masks = true_masks.unsqueeze(1)

            with torch.no_grad():
                masks_pred = model(images)
                masks_pred = torch.sigmoid(masks_pred)

            loss = criterion(masks_pred, true_masks)
            val_loss += loss.item()
            examples_so_far += 1

            dice = dice_loss(masks_pred, true_masks)
            if with_dice:
                loss += dice
            val_dice += (1. - dice.item())

            iou = IoU_loss(masks_pred, true_masks)
            if with_iou:
                loss += iou
            val_iou += (1. - iou.item())

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}/{epochs} - Validation Loss: {val_loss / examples_so_far:.3f}, IoU: {val_iou / examples_so_far:.3f}, Dice: {val_dice / examples_so_far:.3f}")

    return {
        "validation_loss": val_loss / examples_so_far,
        "validation_DICE_score": val_dice / examples_so_far,
        "validation_IoU_score": val_iou / examples_so_far,
    }

def train_model(model, device, train_dataloader, val_dataloader, epochs=10, lr=1e-4, with_dice=True, with_iou=True):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    results = []

    for epoch in range(1, epochs + 1):
        train_loss = 0
        train_dice = 0
        train_iou = 0
        examples_so_far = 0

        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} - Training Loss: 0") as pbar:
            for batch in train_dataloader:
                optimizer.zero_grad()
                images, true_masks = batch[0].to(device), batch[1].to(device).float()

                true_masks = true_masks.unsqueeze(1)

                masks_pred = model(images)
                masks_pred = torch.sigmoid(masks_pred)

                loss = criterion(masks_pred, true_masks)

                if with_dice:
                    dice = dice_loss(masks_pred, true_masks)
                    loss += dice
                    train_dice += (1. - dice.item())

                if with_iou:
                    iou = IoU_loss(masks_pred, true_masks)
                    loss += iou
                    train_iou += (1. - iou.item())

                accelerator.backward(loss)
                optimizer.step()

                train_loss += loss.item()
                examples_so_far += 1

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch}/{epochs} - Training Loss: {train_loss / examples_so_far:.3f}")

        epoch_result = {
            "training_loss": train_loss / examples_so_far,
            "training_DICE_score": train_dice / examples_so_far if with_dice else None,
            "training_IoU_score": train_iou / examples_so_far if with_iou else None,
        }

        val_result = evaluate_model(model, val_dataloader, epoch, epochs, criterion, with_dice, with_iou)
        epoch_result.update(val_result)
        results.append(epoch_result)

    return results


def plot_training_progress(results, save_path=None):

    epochs = range(1, len(results) + 1)
    train_loss = [res["training_loss"] for res in results]
    val_loss = [res["validation_loss"] for res in results]

    train_dice = [res.get("training_DICE_score", None) for res in results]
    val_dice = [res["validation_DICE_score"] for res in results]

    train_iou = [res.get("training_IoU_score", None) for res in results]
    val_iou = [res["validation_IoU_score"] for res in results]

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", linestyle="-", color="red")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o", linestyle="--", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_dice, label="Train Dice", marker="o", linestyle="-", color="green")
    plt.plot(epochs, val_dice, label="Validation Dice", marker="o", linestyle="--", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Training vs Validation Dice Score")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_iou, label="Train IoU", marker="o", linestyle="-", color="orange")
    plt.plot(epochs, val_iou, label="Validation IoU", marker="o", linestyle="--", color="cyan")
    plt.xlabel("Epochs")
    plt.ylabel("IoU Score")
    plt.title("Training vs Validation IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def show_inference(batch, predictions):
    batch_size = batch[0].shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4.*batch_size), squeeze=True, sharey=True, sharex=True)
    fig.subplots_adjust(hspace=0.05, wspace=0)

    for i in range(batch_size):
        img, mask = batch[0][i], batch[1][i]

        axes[i, 0].imshow(decode_image(img.permute(1,2, 0)))
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title("Input Image")

        axes[i, 1].imshow(mask.squeeze(), cmap="gray")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        if i == 0:
            axes[i, 1].set_title("True Mask")

        predicted = predictions[i].squeeze(0)
        predicted = (predicted > 0.5).float()

        axes[i, 2].imshow(predicted.cpu(), cmap="gray")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        if i == 0:
            axes[i, 2].set_title("Predicted Mask")

    plt.show()


"""
##Unet
"""

class ConvBlock(nn.Module):
    """apply twice convolution followed by batch normalization and relu. Preserves the width and height of input"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.cn1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.activ1 = nn.ReLU(inplace=True)
        self.cn2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activ2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.activ1(x)
        x = self.cn2(x)
        x = self.bn2(x)
        return self.activ2(x)

class DownScale(nn.Module):
    """Downscaling with maxpool then ConvBlock, transforming an input with (h, w, in_channels) to (h/2, w/2, out_channels)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x

class UpScale(nn.Module):
    """apply upscaling and then convolution block transforming an input with (h,w,in_channels) to (2h, 2w, out_channels).
       Forward function also simplifies Unet propagation by taking two inputs : first one from constantly propagating (from upscaling)
       and the second one, which is the output from applying Downscale (first input is upscaled, then concatenated with second)"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is (batch, channel, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, start=16, bilinear=False):
        if n_classes == 2:
          n_classes = 1
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBlock(n_channels, start)
        self.down1 = DownScale(start, 2*start)
        self.down2 = DownScale(2*start, 4*start)
        self.down3 = DownScale(4*start, 8*start)

        factor = 2 if bilinear else 1
        self.down4 = DownScale(8*start, 16*start // factor)

        self.up1 = UpScale(16*start, 8*start // factor, bilinear)
        self.up2 = UpScale(8*start, 4*start // factor, bilinear)
        self.up3 = UpScale(4*start, 2*start // factor, bilinear)
        self.up4 = UpScale(2*start, start, bilinear)
        self.outc = nn.Conv2d(start, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

number_of_classes = 2
summary(Unet(3, number_of_classes), input_data=eval_batch_data[0])
# Output:
#   ==========================================================================================

#   Layer (type:depth-idx)                   Output Shape              Param #

#   ==========================================================================================

#   Unet                                     [8, 1, 224, 224]          --

#   ├─ConvBlock: 1-1                         [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-1                       [8, 16, 224, 224]         432

#   │    └─BatchNorm2d: 2-2                  [8, 16, 224, 224]         32

#   │    └─ReLU: 2-3                         [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-4                       [8, 16, 224, 224]         2,304

#   │    └─BatchNorm2d: 2-5                  [8, 16, 224, 224]         32

#   │    └─ReLU: 2-6                         [8, 16, 224, 224]         --

#   ├─DownScale: 1-2                         [8, 32, 112, 112]         --

#   │    └─MaxPool2d: 2-7                    [8, 16, 112, 112]         --

#   │    └─ConvBlock: 2-8                    [8, 32, 112, 112]         --

#   │    │    └─Conv2d: 3-1                  [8, 32, 112, 112]         4,608

#   │    │    └─BatchNorm2d: 3-2             [8, 32, 112, 112]         64

#   │    │    └─ReLU: 3-3                    [8, 32, 112, 112]         --

#   │    │    └─Conv2d: 3-4                  [8, 32, 112, 112]         9,216

#   │    │    └─BatchNorm2d: 3-5             [8, 32, 112, 112]         64

#   │    │    └─ReLU: 3-6                    [8, 32, 112, 112]         --

#   ├─DownScale: 1-3                         [8, 64, 56, 56]           --

#   │    └─MaxPool2d: 2-9                    [8, 32, 56, 56]           --

#   │    └─ConvBlock: 2-10                   [8, 64, 56, 56]           --

#   │    │    └─Conv2d: 3-7                  [8, 64, 56, 56]           18,432

#   │    │    └─BatchNorm2d: 3-8             [8, 64, 56, 56]           128

#   │    │    └─ReLU: 3-9                    [8, 64, 56, 56]           --

#   │    │    └─Conv2d: 3-10                 [8, 64, 56, 56]           36,864

#   │    │    └─BatchNorm2d: 3-11            [8, 64, 56, 56]           128

#   │    │    └─ReLU: 3-12                   [8, 64, 56, 56]           --

#   ├─DownScale: 1-4                         [8, 128, 28, 28]          --

#   │    └─MaxPool2d: 2-11                   [8, 64, 28, 28]           --

#   │    └─ConvBlock: 2-12                   [8, 128, 28, 28]          --

#   │    │    └─Conv2d: 3-13                 [8, 128, 28, 28]          73,728

#   │    │    └─BatchNorm2d: 3-14            [8, 128, 28, 28]          256

#   │    │    └─ReLU: 3-15                   [8, 128, 28, 28]          --

#   │    │    └─Conv2d: 3-16                 [8, 128, 28, 28]          147,456

#   │    │    └─BatchNorm2d: 3-17            [8, 128, 28, 28]          256

#   │    │    └─ReLU: 3-18                   [8, 128, 28, 28]          --

#   ├─DownScale: 1-5                         [8, 256, 14, 14]          --

#   │    └─MaxPool2d: 2-13                   [8, 128, 14, 14]          --

#   │    └─ConvBlock: 2-14                   [8, 256, 14, 14]          --

#   │    │    └─Conv2d: 3-19                 [8, 256, 14, 14]          294,912

#   │    │    └─BatchNorm2d: 3-20            [8, 256, 14, 14]          512

#   │    │    └─ReLU: 3-21                   [8, 256, 14, 14]          --

#   │    │    └─Conv2d: 3-22                 [8, 256, 14, 14]          589,824

#   │    │    └─BatchNorm2d: 3-23            [8, 256, 14, 14]          512

#   │    │    └─ReLU: 3-24                   [8, 256, 14, 14]          --

#   ├─UpScale: 1-6                           [8, 128, 28, 28]          --

#   │    └─ConvTranspose2d: 2-15             [8, 128, 28, 28]          131,200

#   │    └─ConvBlock: 2-16                   [8, 128, 28, 28]          --

#   │    │    └─Conv2d: 3-25                 [8, 128, 28, 28]          294,912

#   │    │    └─BatchNorm2d: 3-26            [8, 128, 28, 28]          256

#   │    │    └─ReLU: 3-27                   [8, 128, 28, 28]          --

#   │    │    └─Conv2d: 3-28                 [8, 128, 28, 28]          147,456

#   │    │    └─BatchNorm2d: 3-29            [8, 128, 28, 28]          256

#   │    │    └─ReLU: 3-30                   [8, 128, 28, 28]          --

#   ├─UpScale: 1-7                           [8, 64, 56, 56]           --

#   │    └─ConvTranspose2d: 2-17             [8, 64, 56, 56]           32,832

#   │    └─ConvBlock: 2-18                   [8, 64, 56, 56]           --

#   │    │    └─Conv2d: 3-31                 [8, 64, 56, 56]           73,728

#   │    │    └─BatchNorm2d: 3-32            [8, 64, 56, 56]           128

#   │    │    └─ReLU: 3-33                   [8, 64, 56, 56]           --

#   │    │    └─Conv2d: 3-34                 [8, 64, 56, 56]           36,864

#   │    │    └─BatchNorm2d: 3-35            [8, 64, 56, 56]           128

#   │    │    └─ReLU: 3-36                   [8, 64, 56, 56]           --

#   ├─UpScale: 1-8                           [8, 32, 112, 112]         --

#   │    └─ConvTranspose2d: 2-19             [8, 32, 112, 112]         8,224

#   │    └─ConvBlock: 2-20                   [8, 32, 112, 112]         --

#   │    │    └─Conv2d: 3-37                 [8, 32, 112, 112]         18,432

#   │    │    └─BatchNorm2d: 3-38            [8, 32, 112, 112]         64

#   │    │    └─ReLU: 3-39                   [8, 32, 112, 112]         --

#   │    │    └─Conv2d: 3-40                 [8, 32, 112, 112]         9,216

#   │    │    └─BatchNorm2d: 3-41            [8, 32, 112, 112]         64

#   │    │    └─ReLU: 3-42                   [8, 32, 112, 112]         --

#   ├─UpScale: 1-9                           [8, 16, 224, 224]         --

#   │    └─ConvTranspose2d: 2-21             [8, 16, 224, 224]         2,064

#   │    └─ConvBlock: 2-22                   [8, 16, 224, 224]         --

#   │    │    └─Conv2d: 3-43                 [8, 16, 224, 224]         4,608

#   │    │    └─BatchNorm2d: 3-44            [8, 16, 224, 224]         32

#   │    │    └─ReLU: 3-45                   [8, 16, 224, 224]         --

#   │    │    └─Conv2d: 3-46                 [8, 16, 224, 224]         2,304

#   │    │    └─BatchNorm2d: 3-47            [8, 16, 224, 224]         32

#   │    │    └─ReLU: 3-48                   [8, 16, 224, 224]         --

#   ├─Conv2d: 1-10                           [8, 1, 224, 224]          17

#   ==========================================================================================

#   Total params: 1,942,577

#   Trainable params: 1,942,577

#   Non-trainable params: 0

#   Total mult-adds (Units.GIGABYTES): 21.05

#   ==========================================================================================

#   Input size (MB): 4.82

#   Forward/backward pass size (MB): 883.10

#   Params size (MB): 7.77

#   Estimated Total Size (MB): 895.68

#   ==========================================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(3, number_of_classes)
model = model.to(device)
Unet_training_val_summary = train_model(model, device, ebhi_train_dataloader, ebhi_val_dataloader,
                                        lr=2e-4, epochs=cfg.SINGLE_NETWORK_TRAINING_EPOCHS,with_dice=True, with_iou=True)
# Output:
#   Epoch 1/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 1/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 2/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 2/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 3/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 3/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 4/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 4/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 5/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 5/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 6/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 6/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 7/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 7/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 8/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 8/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 9/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 9/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 10/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 10/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 11/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 11/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 12/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 12/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 13/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 13/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 14/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 14/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 15/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 15/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 16/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 16/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 17/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 17/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 18/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 18/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 19/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 19/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 20/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 20/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]

plot_training_progress(Unet_training_val_summary)
# Output:
#   <Figure size 2000x600 with 3 Axes>

batch = next(iter(ebhi_val_dataloader))
predictions = torch.sigmoid(model(batch[0].to(device)))
show_inference(batch, predictions)
# Output:
#   <Figure size 1200x3200 with 24 Axes>

"""
##Nested Unet
"""

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

summary(NestedUNet(number_of_classes), input_data=eval_batch_data[0])
# Output:
#   ==========================================================================================

#   Layer (type:depth-idx)                   Output Shape              Param #

#   ==========================================================================================

#   NestedUNet                               [8, 1, 224, 224]          --

#   ├─VGGBlock: 1-1                          [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-1                       [8, 16, 224, 224]         448

#   │    └─BatchNorm2d: 2-2                  [8, 16, 224, 224]         32

#   │    └─ReLU: 2-3                         [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-4                       [8, 16, 224, 224]         2,320

#   │    └─BatchNorm2d: 2-5                  [8, 16, 224, 224]         32

#   │    └─ReLU: 2-6                         [8, 16, 224, 224]         --

#   ├─MaxPool2d: 1-2                         [8, 16, 112, 112]         --

#   ├─VGGBlock: 1-3                          [8, 32, 112, 112]         --

#   │    └─Conv2d: 2-7                       [8, 32, 112, 112]         4,640

#   │    └─BatchNorm2d: 2-8                  [8, 32, 112, 112]         64

#   │    └─ReLU: 2-9                         [8, 32, 112, 112]         --

#   │    └─Conv2d: 2-10                      [8, 32, 112, 112]         9,248

#   │    └─BatchNorm2d: 2-11                 [8, 32, 112, 112]         64

#   │    └─ReLU: 2-12                        [8, 32, 112, 112]         --

#   ├─Upsample: 1-4                          [8, 32, 224, 224]         --

#   ├─VGGBlock: 1-5                          [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-13                      [8, 16, 224, 224]         6,928

#   │    └─BatchNorm2d: 2-14                 [8, 16, 224, 224]         32

#   │    └─ReLU: 2-15                        [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-16                      [8, 16, 224, 224]         2,320

#   │    └─BatchNorm2d: 2-17                 [8, 16, 224, 224]         32

#   │    └─ReLU: 2-18                        [8, 16, 224, 224]         --

#   ├─MaxPool2d: 1-6                         [8, 32, 56, 56]           --

#   ├─VGGBlock: 1-7                          [8, 64, 56, 56]           --

#   │    └─Conv2d: 2-19                      [8, 64, 56, 56]           18,496

#   │    └─BatchNorm2d: 2-20                 [8, 64, 56, 56]           128

#   │    └─ReLU: 2-21                        [8, 64, 56, 56]           --

#   │    └─Conv2d: 2-22                      [8, 64, 56, 56]           36,928

#   │    └─BatchNorm2d: 2-23                 [8, 64, 56, 56]           128

#   │    └─ReLU: 2-24                        [8, 64, 56, 56]           --

#   ├─Upsample: 1-8                          [8, 64, 112, 112]         --

#   ├─VGGBlock: 1-9                          [8, 32, 112, 112]         --

#   │    └─Conv2d: 2-25                      [8, 32, 112, 112]         27,680

#   │    └─BatchNorm2d: 2-26                 [8, 32, 112, 112]         64

#   │    └─ReLU: 2-27                        [8, 32, 112, 112]         --

#   │    └─Conv2d: 2-28                      [8, 32, 112, 112]         9,248

#   │    └─BatchNorm2d: 2-29                 [8, 32, 112, 112]         64

#   │    └─ReLU: 2-30                        [8, 32, 112, 112]         --

#   ├─Upsample: 1-10                         [8, 32, 224, 224]         --

#   ├─VGGBlock: 1-11                         [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-31                      [8, 16, 224, 224]         9,232

#   │    └─BatchNorm2d: 2-32                 [8, 16, 224, 224]         32

#   │    └─ReLU: 2-33                        [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-34                      [8, 16, 224, 224]         2,320

#   │    └─BatchNorm2d: 2-35                 [8, 16, 224, 224]         32

#   │    └─ReLU: 2-36                        [8, 16, 224, 224]         --

#   ├─MaxPool2d: 1-12                        [8, 64, 28, 28]           --

#   ├─VGGBlock: 1-13                         [8, 128, 28, 28]          --

#   │    └─Conv2d: 2-37                      [8, 128, 28, 28]          73,856

#   │    └─BatchNorm2d: 2-38                 [8, 128, 28, 28]          256

#   │    └─ReLU: 2-39                        [8, 128, 28, 28]          --

#   │    └─Conv2d: 2-40                      [8, 128, 28, 28]          147,584

#   │    └─BatchNorm2d: 2-41                 [8, 128, 28, 28]          256

#   │    └─ReLU: 2-42                        [8, 128, 28, 28]          --

#   ├─Upsample: 1-14                         [8, 128, 56, 56]          --

#   ├─VGGBlock: 1-15                         [8, 64, 56, 56]           --

#   │    └─Conv2d: 2-43                      [8, 64, 56, 56]           110,656

#   │    └─BatchNorm2d: 2-44                 [8, 64, 56, 56]           128

#   │    └─ReLU: 2-45                        [8, 64, 56, 56]           --

#   │    └─Conv2d: 2-46                      [8, 64, 56, 56]           36,928

#   │    └─BatchNorm2d: 2-47                 [8, 64, 56, 56]           128

#   │    └─ReLU: 2-48                        [8, 64, 56, 56]           --

#   ├─Upsample: 1-16                         [8, 64, 112, 112]         --

#   ├─VGGBlock: 1-17                         [8, 32, 112, 112]         --

#   │    └─Conv2d: 2-49                      [8, 32, 112, 112]         36,896

#   │    └─BatchNorm2d: 2-50                 [8, 32, 112, 112]         64

#   │    └─ReLU: 2-51                        [8, 32, 112, 112]         --

#   │    └─Conv2d: 2-52                      [8, 32, 112, 112]         9,248

#   │    └─BatchNorm2d: 2-53                 [8, 32, 112, 112]         64

#   │    └─ReLU: 2-54                        [8, 32, 112, 112]         --

#   ├─Upsample: 1-18                         [8, 32, 224, 224]         --

#   ├─VGGBlock: 1-19                         [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-55                      [8, 16, 224, 224]         11,536

#   │    └─BatchNorm2d: 2-56                 [8, 16, 224, 224]         32

#   │    └─ReLU: 2-57                        [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-58                      [8, 16, 224, 224]         2,320

#   │    └─BatchNorm2d: 2-59                 [8, 16, 224, 224]         32

#   │    └─ReLU: 2-60                        [8, 16, 224, 224]         --

#   ├─MaxPool2d: 1-20                        [8, 128, 14, 14]          --

#   ├─VGGBlock: 1-21                         [8, 256, 14, 14]          --

#   │    └─Conv2d: 2-61                      [8, 256, 14, 14]          295,168

#   │    └─BatchNorm2d: 2-62                 [8, 256, 14, 14]          512

#   │    └─ReLU: 2-63                        [8, 256, 14, 14]          --

#   │    └─Conv2d: 2-64                      [8, 256, 14, 14]          590,080

#   │    └─BatchNorm2d: 2-65                 [8, 256, 14, 14]          512

#   │    └─ReLU: 2-66                        [8, 256, 14, 14]          --

#   ├─Upsample: 1-22                         [8, 256, 28, 28]          --

#   ├─VGGBlock: 1-23                         [8, 128, 28, 28]          --

#   │    └─Conv2d: 2-67                      [8, 128, 28, 28]          442,496

#   │    └─BatchNorm2d: 2-68                 [8, 128, 28, 28]          256

#   │    └─ReLU: 2-69                        [8, 128, 28, 28]          --

#   │    └─Conv2d: 2-70                      [8, 128, 28, 28]          147,584

#   │    └─BatchNorm2d: 2-71                 [8, 128, 28, 28]          256

#   │    └─ReLU: 2-72                        [8, 128, 28, 28]          --

#   ├─Upsample: 1-24                         [8, 128, 56, 56]          --

#   ├─VGGBlock: 1-25                         [8, 64, 56, 56]           --

#   │    └─Conv2d: 2-73                      [8, 64, 56, 56]           147,520

#   │    └─BatchNorm2d: 2-74                 [8, 64, 56, 56]           128

#   │    └─ReLU: 2-75                        [8, 64, 56, 56]           --

#   │    └─Conv2d: 2-76                      [8, 64, 56, 56]           36,928

#   │    └─BatchNorm2d: 2-77                 [8, 64, 56, 56]           128

#   │    └─ReLU: 2-78                        [8, 64, 56, 56]           --

#   ├─Upsample: 1-26                         [8, 64, 112, 112]         --

#   ├─VGGBlock: 1-27                         [8, 32, 112, 112]         --

#   │    └─Conv2d: 2-79                      [8, 32, 112, 112]         46,112

#   │    └─BatchNorm2d: 2-80                 [8, 32, 112, 112]         64

#   │    └─ReLU: 2-81                        [8, 32, 112, 112]         --

#   │    └─Conv2d: 2-82                      [8, 32, 112, 112]         9,248

#   │    └─BatchNorm2d: 2-83                 [8, 32, 112, 112]         64

#   │    └─ReLU: 2-84                        [8, 32, 112, 112]         --

#   ├─Upsample: 1-28                         [8, 32, 224, 224]         --

#   ├─VGGBlock: 1-29                         [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-85                      [8, 16, 224, 224]         13,840

#   │    └─BatchNorm2d: 2-86                 [8, 16, 224, 224]         32

#   │    └─ReLU: 2-87                        [8, 16, 224, 224]         --

#   │    └─Conv2d: 2-88                      [8, 16, 224, 224]         2,320

#   │    └─BatchNorm2d: 2-89                 [8, 16, 224, 224]         32

#   │    └─ReLU: 2-90                        [8, 16, 224, 224]         --

#   ├─Conv2d: 1-30                           [8, 1, 224, 224]          17

#   ==========================================================================================

#   Total params: 2,293,793

#   Trainable params: 2,293,793

#   Non-trainable params: 0

#   Total mult-adds (Units.GIGABYTES): 53.00

#   ==========================================================================================

#   Input size (MB): 4.82

#   Forward/backward pass size (MB): 1660.22

#   Params size (MB): 9.18

#   Estimated Total Size (MB): 1674.22

#   ==========================================================================================

model_nested = NestedUNet(number_of_classes)
model_nested = model_nested.to(device)
nested_Unet_training_val_summary = train_model(model_nested, device, ebhi_train_dataloader, ebhi_val_dataloader,
                                        lr=2e-4, epochs=cfg.SINGLE_NETWORK_TRAINING_EPOCHS,with_dice=True, with_iou=True)

# Output:
#   Epoch 1/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 1/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 2/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 2/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 3/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 3/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 4/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 4/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 5/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 5/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 6/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 6/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 7/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 7/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 8/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 8/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 9/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 9/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 10/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 10/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 11/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 11/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 12/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 12/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 13/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 13/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 14/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 14/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 15/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 15/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 16/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 16/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 17/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 17/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 18/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 18/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 19/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 19/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 20/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 20/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]

plot_training_progress(nested_Unet_training_val_summary)
# Output:
#   <Figure size 2000x600 with 3 Axes>

batch = next(iter(ebhi_val_dataloader))
predictions = torch.sigmoid(model_nested(batch[0].to(device)))
show_inference(batch, predictions)
# Output:
#   <Figure size 1200x3200 with 24 Axes>

"""
##Attention Unet
"""

class ConvBlockAttention(nn.Module):
    """Simplified convolution block with a single Conv layer, BatchNorm, and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownScale(nn.Module):
    """Downscaling with maxpool then ConvBlock, transforming an input with (h, w, in_channels) to (h/2, w/2, out_channels)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlockAttention(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x

class AttentionBlock(nn.Module):
    """Simplified Attention Gate."""
    def __init__(self, f_g, f_l, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv_g = nn.Conv2d(f_g, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_x = nn.Conv2d(f_l, out_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        psi = self.sigmoid(self.psi(F.relu(self.conv_g(g) + self.conv_x(x))))
        return x * psi

class UpScale(nn.Module):
    """Upscaling with a single ConvBlock and an optional Attention Gate."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ConvBlockAttention(in_channels, out_channels)
        self.attention = AttentionBlock(f_g=in_channels // 2, f_l=out_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.attention(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionUnetLite(nn.Module):
    """Lightweight Attention U-Net with reduced depth and complexity."""
    def __init__(self, n_channels, n_classes, start=32, bilinear=False):
        super(AttentionUnetLite, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBlockAttention(n_channels, start)
        self.down1 = DownScale(start, 2*start)
        self.down2 = DownScale(2*start, 4*start)

        factor = 2 if bilinear else 1
        self.down3 = DownScale(4*start, 8*start // factor)

        self.up1 = UpScale(8*start, 4*start // factor, bilinear)
        self.up2 = UpScale(4*start, 2*start, bilinear)
        self.up3 = UpScale(2*start, start, bilinear)
        self.outc = nn.Conv2d(start, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


summary(AttentionUnetLite(3, number_of_classes), input_data=eval_batch_data[0])
# Output:
#   ==========================================================================================

#   Layer (type:depth-idx)                   Output Shape              Param #

#   ==========================================================================================

#   AttentionUnetLite                        [8, 1, 224, 224]          --

#   ├─ConvBlockAttention: 1-1                [8, 32, 224, 224]         --

#   │    └─Conv2d: 2-1                       [8, 32, 224, 224]         864

#   │    └─BatchNorm2d: 2-2                  [8, 32, 224, 224]         64

#   │    └─ReLU: 2-3                         [8, 32, 224, 224]         --

#   ├─DownScale: 1-2                         [8, 64, 112, 112]         --

#   │    └─MaxPool2d: 2-4                    [8, 32, 112, 112]         --

#   │    └─ConvBlockAttention: 2-5           [8, 64, 112, 112]         --

#   │    │    └─Conv2d: 3-1                  [8, 64, 112, 112]         18,432

#   │    │    └─BatchNorm2d: 3-2             [8, 64, 112, 112]         128

#   │    │    └─ReLU: 3-3                    [8, 64, 112, 112]         --

#   ├─DownScale: 1-3                         [8, 128, 56, 56]          --

#   │    └─MaxPool2d: 2-6                    [8, 64, 56, 56]           --

#   │    └─ConvBlockAttention: 2-7           [8, 128, 56, 56]          --

#   │    │    └─Conv2d: 3-4                  [8, 128, 56, 56]          73,728

#   │    │    └─BatchNorm2d: 3-5             [8, 128, 56, 56]          256

#   │    │    └─ReLU: 3-6                    [8, 128, 56, 56]          --

#   ├─DownScale: 1-4                         [8, 256, 28, 28]          --

#   │    └─MaxPool2d: 2-8                    [8, 128, 28, 28]          --

#   │    └─ConvBlockAttention: 2-9           [8, 256, 28, 28]          --

#   │    │    └─Conv2d: 3-7                  [8, 256, 28, 28]          294,912

#   │    │    └─BatchNorm2d: 3-8             [8, 256, 28, 28]          512

#   │    │    └─ReLU: 3-9                    [8, 256, 28, 28]          --

#   ├─UpScale: 1-5                           [8, 128, 56, 56]          --

#   │    └─ConvTranspose2d: 2-10             [8, 128, 56, 56]          131,200

#   │    └─AttentionBlock: 2-11              [8, 128, 56, 56]          --

#   │    │    └─Conv2d: 3-10                 [8, 128, 56, 56]          16,512

#   │    │    └─Conv2d: 3-11                 [8, 128, 56, 56]          16,512

#   │    │    └─Conv2d: 3-12                 [8, 1, 56, 56]            129

#   │    │    └─Sigmoid: 3-13                [8, 1, 56, 56]            --

#   │    └─ConvBlockAttention: 2-12          [8, 128, 56, 56]          --

#   │    │    └─Conv2d: 3-14                 [8, 128, 56, 56]          294,912

#   │    │    └─BatchNorm2d: 3-15            [8, 128, 56, 56]          256

#   │    │    └─ReLU: 3-16                   [8, 128, 56, 56]          --

#   ├─UpScale: 1-6                           [8, 64, 112, 112]         --

#   │    └─ConvTranspose2d: 2-13             [8, 64, 112, 112]         32,832

#   │    └─AttentionBlock: 2-14              [8, 64, 112, 112]         --

#   │    │    └─Conv2d: 3-17                 [8, 64, 112, 112]         4,160

#   │    │    └─Conv2d: 3-18                 [8, 64, 112, 112]         4,160

#   │    │    └─Conv2d: 3-19                 [8, 1, 112, 112]          65

#   │    │    └─Sigmoid: 3-20                [8, 1, 112, 112]          --

#   │    └─ConvBlockAttention: 2-15          [8, 64, 112, 112]         --

#   │    │    └─Conv2d: 3-21                 [8, 64, 112, 112]         73,728

#   │    │    └─BatchNorm2d: 3-22            [8, 64, 112, 112]         128

#   │    │    └─ReLU: 3-23                   [8, 64, 112, 112]         --

#   ├─UpScale: 1-7                           [8, 32, 224, 224]         --

#   │    └─ConvTranspose2d: 2-16             [8, 32, 224, 224]         8,224

#   │    └─AttentionBlock: 2-17              [8, 32, 224, 224]         --

#   │    │    └─Conv2d: 3-24                 [8, 32, 224, 224]         1,056

#   │    │    └─Conv2d: 3-25                 [8, 32, 224, 224]         1,056

#   │    │    └─Conv2d: 3-26                 [8, 1, 224, 224]          33

#   │    │    └─Sigmoid: 3-27                [8, 1, 224, 224]          --

#   │    └─ConvBlockAttention: 2-18          [8, 32, 224, 224]         --

#   │    │    └─Conv2d: 3-28                 [8, 32, 224, 224]         18,432

#   │    │    └─BatchNorm2d: 3-29            [8, 32, 224, 224]         64

#   │    │    └─ReLU: 3-30                   [8, 32, 224, 224]         --

#   ├─Conv2d: 1-8                            [8, 1, 224, 224]          33

#   ==========================================================================================

#   Total params: 992,388

#   Trainable params: 992,388

#   Non-trainable params: 0

#   Total mult-adds (Units.GIGABYTES): 40.53

#   ==========================================================================================

#   Input size (MB): 4.82

#   Forward/backward pass size (MB): 1291.93

#   Params size (MB): 3.97

#   Estimated Total Size (MB): 1300.72

#   ==========================================================================================

model = AttentionUnetLite(3, number_of_classes)
model = model.to(device)
AttentionUnet_training_val_summary = train_model(model, device, ebhi_train_dataloader, ebhi_val_dataloader,
                                        lr=2e-4, epochs=cfg.SINGLE_NETWORK_TRAINING_EPOCHS,with_dice=True, with_iou=True)
# Output:
#   Epoch 1/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 1/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 2/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 2/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 3/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 3/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 4/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 4/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 5/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 5/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 6/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 6/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 7/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 7/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 8/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 8/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 9/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 9/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 10/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 10/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 11/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 11/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 12/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 12/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 13/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 13/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 14/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 14/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 15/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 15/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 16/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 16/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 17/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 17/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 18/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 18/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 19/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 19/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]
#   Epoch 20/20 - Training Loss: 0:   0%|          | 0/192 [00:00<?, ?it/s]
#   Epoch 20/20 - Validation Loss: 0:   0%|          | 0/83 [00:00<?, ?it/s]

plot_training_progress(AttentionUnet_training_val_summary)
# Output:
#   <Figure size 2000x600 with 3 Axes>

batch = next(iter(ebhi_val_dataloader))
predictions = torch.sigmoid(model(batch[0].to(device)))
show_inference(batch, predictions)
# Output:
#   <Figure size 1200x3200 with 24 Axes>



================================================
FILE: Multiclass_Segmentation.ipynb
================================================
# Jupyter notebook converted to Python script.

from google.colab import drive
drive.mount('/content/drive')
# Output:
#   Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from accelerate import Accelerator
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from IPython.display import clear_output
from tqdm.auto import tqdm
from time import time
from typing import Tuple
import sys

class CONFIG:

    USE_MIXED_PRECISION = "fp16"
    DOWNSCALE = 2
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    EXTRA_LOSS_EPS = 1e-6
    BATCH_SIZE = 8
    SINGLE_NETWORK_TRAINING_EPOCHS = 15
    CE_VS_DICE_EVAL_EPOCHS = 15
    DELTA_BETA = 0.2

cfg = CONFIG()

datapath = os.path.join("/content/drive/MyDrive", "cityscapes_data")

train_datapath = os.path.join(datapath, "train")
val_datapath = os.path.join(datapath, "val")
train_cs_datapath = os.path.join(datapath, "cityscapes_data", "train")
val_cs_datapath = os.path.join(datapath, "cityscapes_data", "val")
training_images_paths = [os.path.join(train_datapath, f) for f in os.listdir(train_datapath)]
validation_images_paths = [os.path.join(val_datapath, f) for f in os.listdir(val_datapath)]


print(f"size of training : {len(training_images_paths)}")
print(f"size of validation : {len(validation_images_paths)}")
# Output:
#   size of training : 2975

#   size of validation : 500


if cfg.USE_MIXED_PRECISION is not None:
    accelerator = Accelerator(mixed_precision=cfg.USE_MIXED_PRECISION)
else:
    accelerator = Accelerator()

width = 2
height = 2
vis_batch_size = width * height
indexes = np.arange(len(training_images_paths))
indexes = np.random.permutation(indexes)[:vis_batch_size]


fig, axs = plt.subplots(height, width, sharex=True, sharey=True, figsize=(20, 10))
for i in range(vis_batch_size):

    img = torchvision.io.read_image(training_images_paths[indexes[i]])
    img = img.permute(1, 2, 0)
    y, x = i // width, i % width
    axs[y, x].imshow(img.numpy())

plt.tight_layout()
# Output:
#   <Figure size 2000x1000 with 4 Axes>

# link : https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
idx_to_name = [ 'unlabeled','ego vehicle','rectification border', 'out of roi', 'static', 'dynamic','ground', 'road',
               'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence','guard rail' , 'bridge','tunnel','pole',
               'polegroup', 'traffic light', 'traffic sign' ,'vegetation', 'terrain', 'sky' ,'person', 'rider', 'car',
               'truck' ,'bus', 'caravan','trailer', 'train' , 'motorcycle','bicycle','license plate']

idx_to_category = ["void", "flat", "construction", "object", "nature", "sky", "human", "vehicle"]

idx_to_color = [[ 0,  0,  0], [ 0,  0,  0], [  0,  0,  0], [  0,  0,  0],[ 0,  0,  0],[111, 74,  0],[81,  0, 81] ,[128, 64,128],[244, 35,232],
                [250,170,160],[230,150,140],[70, 70, 70],[102,102,156],[190,153,153],[180,165,180],[150,100,100],[150,120, 90],[153,153,153],
                [153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180],[220, 20, 60],[255,  0,  0],[ 0,  0,142],
                [ 0,  0, 70],[ 0, 60,100],[ 0,  0, 90],[  0,  0,110],[ 0, 80,100],[  0,  0,230],[119, 11, 32],[  0,  0,142]]


idx_to_color_np = np.array(idx_to_color)

name_to_category = {0 : 0, 1 : 0, 2 : 0, 3: 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1, 8 : 1, 9 : 1, 10 : 1, 11 :2, 12 : 2, 13 : 2, 14 : 2, 15 : 2, 16 : 2,
                    17 : 3, 18 : 3, 19 : 3, 20: 3, 21 : 4, 22 : 4, 23 : 5, 24 : 6, 25 : 6, 26 : 7, 27 : 7, 28 : 7, 29 : 7, 30 : 7, 31 : 7, 32: 7, 33 : 7, 34 : 7}


category_colors = [
    [0, 0, 0],        # void (black)
    [128, 64, 128],   # flat (purple)
    [70, 70, 70],     # construction (dark gray)
    [220, 220, 0],    # object (yellow)
    [107, 142, 35],   # nature (green)
    [70, 130, 180],   # sky (blue)
    [220, 20, 60],    # human (red)
    [0, 0, 142]       # vehicle (dark blue)
]
category_colors_norm = np.array(category_colors) / 255.0
category_cmap = mcolors.ListedColormap(category_colors_norm, name="Cityscapes_Categories")
bounds = np.arange(len(idx_to_category) + 1) - 0.5
norm = mcolors.BoundaryNorm(bounds, category_cmap.N)
fig, ax = plt.subplots(figsize=(10, 2))
cb = plt.colorbar(
    plt.cm.ScalarMappable(cmap=category_cmap, norm=norm),
    cax=ax, orientation="horizontal", ticks=np.arange(len(idx_to_category))
)
cb.set_label("Cityscapes Categories")
# Output:
#   <Figure size 1000x200 with 1 Axes>

name_to_category_mapping = lambda x: name_to_category[x]
vectorized_cat_mapping = np.vectorize(name_to_category_mapping)
name_to_col_mapping = lambda x: idx_to_color[x]
vectorized_col_mapping = np.vectorize(name_to_col_mapping)

def preprocess_image(path : str, sparse_mapping=True, downscale_factor=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Read the .jpeg image from *path*. Return the input image (256 x 256 x 3), mask (256 x 256 x 3) read from the jpeg
        and conversion to categories or names (if sparse_mapping is true) representation (256 x 256 x (|categories| or |names|) )
    """

    img = Image.open(path)
    width, height = img.size

    if downscale_factor:
        width, height = width // downscale_factor, height//downscale_factor
        img = img.resize(( width, height ))

    img = np.asarray(img)
    raw, mask = img[:, :width//2, :], img[:, width//2:, :]

    height, width, channels = mask.shape
    distances = np.sum((mask.reshape(-1, channels)[:, np.newaxis, :] - idx_to_color_np)**2, axis=2)
    classes = np.argmin(distances, axis=1).reshape(height, width)
    if sparse_mapping:
        classes = vectorized_cat_mapping(classes)

    return raw, mask, classes

train_images_to_use = -1
downscale_factor=cfg.DOWNSCALE

X_train, Y_train = [], []
X_val, Y_val = [], []

for path in tqdm(training_images_paths[:]):
    X, _, Y = preprocess_image(path, downscale_factor=downscale_factor)
    X_train.append(torch.Tensor(X / 255.).permute(2, 0, 1))
    Y_train.append(torch.Tensor(Y))

for path in tqdm(validation_images_paths):
    X, _, Y = preprocess_image(path, downscale_factor=downscale_factor)
    X_val.append(torch.Tensor(X / 255.).permute(2, 0, 1))
    Y_val.append(torch.Tensor(Y))
# Output:
#     0%|          | 0/2975 [00:00<?, ?it/s]
#     0%|          | 0/500 [00:00<?, ?it/s]

class CityScapesDataset(Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x , y

preprocess = transforms.Compose([
    transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
])


train_ds = CityScapesDataset(X_train, Y_train, transform=preprocess)
val_ds = CityScapesDataset(X_val, Y_val, transform=preprocess)
train_dataloader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)


eval_batch_data = next(iter(val_dataloader))
def decode_image(img : torch.Tensor) -> torch.Tensor:
    return img * torch.Tensor(cfg.STD) + torch.Tensor(cfg.MEAN)

print(eval_batch_data[0].shape, eval_batch_data[1].shape)
batch_size = eval_batch_data[0].shape[0]
fig, axes = plt.subplots(batch_size, 2, figsize=(4, 2.*batch_size), squeeze=True)
fig.subplots_adjust(hspace=0.0, wspace=0.0)

for i in range(batch_size):
    img, mask = eval_batch_data[0][i], eval_batch_data[1][i]
    #print(img.shape, mask.shape)
    axes[i, 0].imshow(decode_image(img.permute(1,2, 0)))
    axes[i,0].set_xticks([])
    axes[i,0].set_yticks([])

    axes[i, 1].imshow(mask, cmap=category_cmap)
    axes[i,1].set_xticks([])
    axes[i,1].set_yticks([])
# Output:
#   torch.Size([8, 3, 128, 128]) torch.Size([8, 128, 128])

#   <Figure size 400x1600 with 16 Axes>

# dice loss
# awesome implementation for DICE can be found here
# https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
def dice_coeff(inp : Tensor, tgt : Tensor, eps=cfg.EXTRA_LOSS_EPS):
    sum_dim = (-1, -2, -3)
    inter = 2 *(inp * tgt).sum(dim=sum_dim)

    # calculate the sum of |inp| + |tgt|
    sets_sum = inp.sum(dim=sum_dim) + tgt.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + eps) / (sets_sum + eps)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, eps: float = cfg.EXTRA_LOSS_EPS):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), eps)

def dice_loss(input: Tensor, target: Tensor):
    return 1 - multiclass_dice_coeff(input, target)

def IoU_coeff(inp : Tensor, tgt : Tensor, eps = 1e-6):
    sum_dim = (-1, -2, -3)
    inter = (inp * tgt).sum(dim=sum_dim)
    sets_sum = inp.sum(dim=sum_dim) + tgt.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # IoU = |A ^ B| / |A \/ B| = |A ^ B| / (|A| + |B| - |A^B|)
    return (inter + eps) / (sets_sum - inter + eps)

def IoU_loss(inp : Tensor, tgt : Tensor):
    return 1 - IoU_coeff(inp.flatten(0,1), tgt.flatten(0,1))

def evaluate_model(model, val_dataloader, epoch, epochs, criterion, with_dice=True, with_iou=True):
    val_loss = 0
    val_dice = 0
    val_iou = 0
    examples_so_far = 0

    model.eval()
    with tqdm(val_dataloader, desc=f"Epoch {epoch}/{epochs} - Validation Loss: 0") as pbar:
        for batch in val_dataloader:
            images, true_masks = batch[0].to(device), batch[1].to(device).long()

            with torch.no_grad():
                masks_pred = model(images)

            loss = criterion(masks_pred, true_masks)
            val_loss += loss.item()
            examples_so_far += 1

            num_classes = masks_pred.shape[1]
            true_masks_onehot = F.one_hot(true_masks, num_classes).permute(0, 3, 1, 2).float()

            dice = dice_loss(F.softmax(masks_pred, dim=1), true_masks_onehot)

            if with_dice:
                loss += dice
            val_dice += (1. - dice.item())

            iou = IoU_loss(F.softmax(masks_pred, dim=1), true_masks_onehot)

            if with_iou:
                loss += iou
            val_iou += (1. - iou.item())

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}/{epochs} - Validation Loss: {val_loss / examples_so_far:.3f}, IoU: {val_iou / examples_so_far:.3f}, Dice: {val_dice / examples_so_far:.3f}")

    return {
        "validation_loss": val_loss / examples_so_far,
        "validation_DICE_score": val_dice / examples_so_far,
        "validation_IoU_score": val_iou / examples_so_far,
    }


def train_model(model, device, train_dataloader, val_dataloader, epochs=10, lr=1e-4, with_dice=True, with_iou=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    results = []

    for epoch in range(1, epochs + 1):
        train_loss = 0
        train_dice = 0
        train_iou = 0
        examples_so_far = 0

        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} - Training Loss: 0") as pbar:
            for batch in train_dataloader:
                optimizer.zero_grad()
                images, true_masks = batch[0].to(device), batch[1].to(device).long()

                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                num_classes = masks_pred.shape[1]
                true_masks_onehot = F.one_hot(true_masks, num_classes).permute(0, 3, 1, 2).float()

                if with_dice:
                    dice = dice_loss(F.softmax(masks_pred, dim=1), true_masks_onehot)
                    loss += dice
                    train_dice += (1. - dice.item())

                if with_iou:
                    iou = IoU_loss(F.softmax(masks_pred, dim=1), true_masks_onehot)
                    loss += iou
                    train_iou += (1. - iou.item())

                accelerator.backward(loss)
                optimizer.step()

                train_loss += loss.item()
                examples_so_far += 1

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch}/{epochs} - Training Loss: {train_loss / examples_so_far:.3f}")

        epoch_result = {
            "training_loss": train_loss / examples_so_far,
            "training_DICE_score": train_dice / examples_so_far if with_dice else None,
            "training_IoU_score": train_iou / examples_so_far if with_iou else None,
        }

        val_result = evaluate_model(model, val_dataloader, epoch, epochs, criterion, with_dice, with_iou)
        epoch_result.update(val_result)
        results.append(epoch_result)

    return results

def plot_training_progress(results, save_path=None):

    epochs = range(1, len(results) + 1)
    train_loss = [res["training_loss"] for res in results]
    val_loss = [res["validation_loss"] for res in results]

    train_dice = [res.get("training_DICE_score", None) for res in results]
    val_dice = [res["validation_DICE_score"] for res in results]

    train_iou = [res.get("training_IoU_score", None) for res in results]
    val_iou = [res["validation_IoU_score"] for res in results]

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", linestyle="-", color="red")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o", linestyle="--", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_dice, label="Train Dice", marker="o", linestyle="-", color="green")
    plt.plot(epochs, val_dice, label="Validation Dice", marker="o", linestyle="--", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Training vs Validation Dice Score")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_iou, label="Train IoU", marker="o", linestyle="-", color="orange")
    plt.plot(epochs, val_iou, label="Validation IoU", marker="o", linestyle="--", color="cyan")
    plt.xlabel("Epochs")
    plt.ylabel("IoU Score")
    plt.title("Training vs Validation IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def show_inference(batch, predictions):

    batch_size = batch[0].shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4.*batch_size), squeeze=True, sharey=True, sharex=True)
    fig.subplots_adjust(hspace=0.05, wspace=0)

    for i in range(batch_size):
        img, mask = batch[0][i], batch[1][i]

        axes[i, 0].imshow(decode_image(img.permute(1,2, 0)))
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
        if i == 0:
            axes[i, 0].set_title("Input Image")

        axes[i, 1].imshow(mask, cmap=category_cmap)
        axes[i,1].set_xticks([])
        axes[i,1].set_yticks([])
        if i == 0:
            axes[i, 1].set_title("True Mask")

        predicted = predictions[i]
        predicted = predicted.permute(1, 2, 0)
        predicted = torch.argmax(predicted, dim=2)

        axes[i, 2].imshow(predicted.cpu(), cmap=category_cmap)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        if i == 0:
            axes[i, 2].set_title("Predicted Mask")


"""
##Unet
"""

class ConvBlock(nn.Module):
    """apply twice convolution followed by batch normalization and relu. Preserves the width and height of input"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.cn1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.activ1 = nn.ReLU(inplace=True)
        self.cn2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activ2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.activ1(x)
        x = self.cn2(x)
        x = self.bn2(x)
        return self.activ2(x)

class DownScale(nn.Module):
    """Downscaling with maxpool then ConvBlock, transforming an input with (h, w, in_channels) to (h/2, w/2, out_channels)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x

class UpScale(nn.Module):
    """apply upscaling and then convolution block transforming an input with (h,w,in_channels) to (2h, 2w, out_channels).
       Forward function also simplifies Unet propagation by taking two inputs : first one from constantly propagating (from upscaling)
       and the second one, which is the output from applying Downscale (first input is upscaled, then concatenated with second)"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is (batch, channel, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, start=16, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBlock(n_channels, start)
        self.down1 = DownScale(start, 2*start)
        self.down2 = DownScale(2*start, 4*start)
        self.down3 = DownScale(4*start, 8*start)

        factor = 2 if bilinear else 1
        self.down4 = DownScale(8*start, 16*start // factor)

        self.up1 = UpScale(16*start, 8*start // factor, bilinear)
        self.up2 = UpScale(8*start, 4*start // factor, bilinear)
        self.up3 = UpScale(4*start, 2*start // factor, bilinear)
        self.up4 = UpScale(2*start, start, bilinear)
        self.outc = nn.Conv2d(start, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

number_of_classes = len(set(name_to_category.values()))
summary(Unet(3, number_of_classes), input_data=eval_batch_data[0])
# Output:
#   ==========================================================================================

#   Layer (type:depth-idx)                   Output Shape              Param #

#   ==========================================================================================

#   Unet                                     [8, 8, 128, 128]          --

#   ├─ConvBlock: 1-1                         [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-1                       [8, 16, 128, 128]         432

#   │    └─BatchNorm2d: 2-2                  [8, 16, 128, 128]         32

#   │    └─ReLU: 2-3                         [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-4                       [8, 16, 128, 128]         2,304

#   │    └─BatchNorm2d: 2-5                  [8, 16, 128, 128]         32

#   │    └─ReLU: 2-6                         [8, 16, 128, 128]         --

#   ├─DownScale: 1-2                         [8, 32, 64, 64]           --

#   │    └─MaxPool2d: 2-7                    [8, 16, 64, 64]           --

#   │    └─ConvBlock: 2-8                    [8, 32, 64, 64]           --

#   │    │    └─Conv2d: 3-1                  [8, 32, 64, 64]           4,608

#   │    │    └─BatchNorm2d: 3-2             [8, 32, 64, 64]           64

#   │    │    └─ReLU: 3-3                    [8, 32, 64, 64]           --

#   │    │    └─Conv2d: 3-4                  [8, 32, 64, 64]           9,216

#   │    │    └─BatchNorm2d: 3-5             [8, 32, 64, 64]           64

#   │    │    └─ReLU: 3-6                    [8, 32, 64, 64]           --

#   ├─DownScale: 1-3                         [8, 64, 32, 32]           --

#   │    └─MaxPool2d: 2-9                    [8, 32, 32, 32]           --

#   │    └─ConvBlock: 2-10                   [8, 64, 32, 32]           --

#   │    │    └─Conv2d: 3-7                  [8, 64, 32, 32]           18,432

#   │    │    └─BatchNorm2d: 3-8             [8, 64, 32, 32]           128

#   │    │    └─ReLU: 3-9                    [8, 64, 32, 32]           --

#   │    │    └─Conv2d: 3-10                 [8, 64, 32, 32]           36,864

#   │    │    └─BatchNorm2d: 3-11            [8, 64, 32, 32]           128

#   │    │    └─ReLU: 3-12                   [8, 64, 32, 32]           --

#   ├─DownScale: 1-4                         [8, 128, 16, 16]          --

#   │    └─MaxPool2d: 2-11                   [8, 64, 16, 16]           --

#   │    └─ConvBlock: 2-12                   [8, 128, 16, 16]          --

#   │    │    └─Conv2d: 3-13                 [8, 128, 16, 16]          73,728

#   │    │    └─BatchNorm2d: 3-14            [8, 128, 16, 16]          256

#   │    │    └─ReLU: 3-15                   [8, 128, 16, 16]          --

#   │    │    └─Conv2d: 3-16                 [8, 128, 16, 16]          147,456

#   │    │    └─BatchNorm2d: 3-17            [8, 128, 16, 16]          256

#   │    │    └─ReLU: 3-18                   [8, 128, 16, 16]          --

#   ├─DownScale: 1-5                         [8, 256, 8, 8]            --

#   │    └─MaxPool2d: 2-13                   [8, 128, 8, 8]            --

#   │    └─ConvBlock: 2-14                   [8, 256, 8, 8]            --

#   │    │    └─Conv2d: 3-19                 [8, 256, 8, 8]            294,912

#   │    │    └─BatchNorm2d: 3-20            [8, 256, 8, 8]            512

#   │    │    └─ReLU: 3-21                   [8, 256, 8, 8]            --

#   │    │    └─Conv2d: 3-22                 [8, 256, 8, 8]            589,824

#   │    │    └─BatchNorm2d: 3-23            [8, 256, 8, 8]            512

#   │    │    └─ReLU: 3-24                   [8, 256, 8, 8]            --

#   ├─UpScale: 1-6                           [8, 128, 16, 16]          --

#   │    └─ConvTranspose2d: 2-15             [8, 128, 16, 16]          131,200

#   │    └─ConvBlock: 2-16                   [8, 128, 16, 16]          --

#   │    │    └─Conv2d: 3-25                 [8, 128, 16, 16]          294,912

#   │    │    └─BatchNorm2d: 3-26            [8, 128, 16, 16]          256

#   │    │    └─ReLU: 3-27                   [8, 128, 16, 16]          --

#   │    │    └─Conv2d: 3-28                 [8, 128, 16, 16]          147,456

#   │    │    └─BatchNorm2d: 3-29            [8, 128, 16, 16]          256

#   │    │    └─ReLU: 3-30                   [8, 128, 16, 16]          --

#   ├─UpScale: 1-7                           [8, 64, 32, 32]           --

#   │    └─ConvTranspose2d: 2-17             [8, 64, 32, 32]           32,832

#   │    └─ConvBlock: 2-18                   [8, 64, 32, 32]           --

#   │    │    └─Conv2d: 3-31                 [8, 64, 32, 32]           73,728

#   │    │    └─BatchNorm2d: 3-32            [8, 64, 32, 32]           128

#   │    │    └─ReLU: 3-33                   [8, 64, 32, 32]           --

#   │    │    └─Conv2d: 3-34                 [8, 64, 32, 32]           36,864

#   │    │    └─BatchNorm2d: 3-35            [8, 64, 32, 32]           128

#   │    │    └─ReLU: 3-36                   [8, 64, 32, 32]           --

#   ├─UpScale: 1-8                           [8, 32, 64, 64]           --

#   │    └─ConvTranspose2d: 2-19             [8, 32, 64, 64]           8,224

#   │    └─ConvBlock: 2-20                   [8, 32, 64, 64]           --

#   │    │    └─Conv2d: 3-37                 [8, 32, 64, 64]           18,432

#   │    │    └─BatchNorm2d: 3-38            [8, 32, 64, 64]           64

#   │    │    └─ReLU: 3-39                   [8, 32, 64, 64]           --

#   │    │    └─Conv2d: 3-40                 [8, 32, 64, 64]           9,216

#   │    │    └─BatchNorm2d: 3-41            [8, 32, 64, 64]           64

#   │    │    └─ReLU: 3-42                   [8, 32, 64, 64]           --

#   ├─UpScale: 1-9                           [8, 16, 128, 128]         --

#   │    └─ConvTranspose2d: 2-21             [8, 16, 128, 128]         2,064

#   │    └─ConvBlock: 2-22                   [8, 16, 128, 128]         --

#   │    │    └─Conv2d: 3-43                 [8, 16, 128, 128]         4,608

#   │    │    └─BatchNorm2d: 3-44            [8, 16, 128, 128]         32

#   │    │    └─ReLU: 3-45                   [8, 16, 128, 128]         --

#   │    │    └─Conv2d: 3-46                 [8, 16, 128, 128]         2,304

#   │    │    └─BatchNorm2d: 3-47            [8, 16, 128, 128]         32

#   │    │    └─ReLU: 3-48                   [8, 16, 128, 128]         --

#   ├─Conv2d: 1-10                           [8, 8, 128, 128]          136

#   ==========================================================================================

#   Total params: 1,942,696

#   Trainable params: 1,942,696

#   Non-trainable params: 0

#   Total mult-adds (Units.GIGABYTES): 6.89

#   ==========================================================================================

#   Input size (MB): 1.57

#   Forward/backward pass size (MB): 295.70

#   Params size (MB): 7.77

#   Estimated Total Size (MB): 305.04

#   ==========================================================================================

device = "cuda"
model = Unet(3, number_of_classes)
model = model.to(device)
Unet_training_val_summary = train_model(model, device, train_dataloader, val_dataloader,
                                        lr=3e-4, epochs=cfg.SINGLE_NETWORK_TRAINING_EPOCHS,with_dice=True, with_iou=True)
# Output:
#   Epoch 1/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 1/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 2/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 2/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 3/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 3/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 4/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 4/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 5/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 5/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 6/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 6/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 7/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 7/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 8/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 8/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 9/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 9/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 10/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 10/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 11/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 11/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 12/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 12/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 13/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 13/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 14/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 14/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 15/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 15/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]

plot_training_progress(Unet_training_val_summary)
# Output:
#   <Figure size 2000x600 with 3 Axes>

batch = next(iter(val_dataloader))
predictions = model(batch[0].to(device))
show_inference(batch, predictions)
# Output:
#   <Figure size 1200x3200 with 24 Axes>

"""
##Nested Unet
"""

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

summary(NestedUNet(number_of_classes), input_data=eval_batch_data[0])
# Output:
#   ==========================================================================================

#   Layer (type:depth-idx)                   Output Shape              Param #

#   ==========================================================================================

#   NestedUNet                               [8, 8, 128, 128]          --

#   ├─VGGBlock: 1-1                          [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-1                       [8, 16, 128, 128]         448

#   │    └─BatchNorm2d: 2-2                  [8, 16, 128, 128]         32

#   │    └─ReLU: 2-3                         [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-4                       [8, 16, 128, 128]         2,320

#   │    └─BatchNorm2d: 2-5                  [8, 16, 128, 128]         32

#   │    └─ReLU: 2-6                         [8, 16, 128, 128]         --

#   ├─MaxPool2d: 1-2                         [8, 16, 64, 64]           --

#   ├─VGGBlock: 1-3                          [8, 32, 64, 64]           --

#   │    └─Conv2d: 2-7                       [8, 32, 64, 64]           4,640

#   │    └─BatchNorm2d: 2-8                  [8, 32, 64, 64]           64

#   │    └─ReLU: 2-9                         [8, 32, 64, 64]           --

#   │    └─Conv2d: 2-10                      [8, 32, 64, 64]           9,248

#   │    └─BatchNorm2d: 2-11                 [8, 32, 64, 64]           64

#   │    └─ReLU: 2-12                        [8, 32, 64, 64]           --

#   ├─Upsample: 1-4                          [8, 32, 128, 128]         --

#   ├─VGGBlock: 1-5                          [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-13                      [8, 16, 128, 128]         6,928

#   │    └─BatchNorm2d: 2-14                 [8, 16, 128, 128]         32

#   │    └─ReLU: 2-15                        [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-16                      [8, 16, 128, 128]         2,320

#   │    └─BatchNorm2d: 2-17                 [8, 16, 128, 128]         32

#   │    └─ReLU: 2-18                        [8, 16, 128, 128]         --

#   ├─MaxPool2d: 1-6                         [8, 32, 32, 32]           --

#   ├─VGGBlock: 1-7                          [8, 64, 32, 32]           --

#   │    └─Conv2d: 2-19                      [8, 64, 32, 32]           18,496

#   │    └─BatchNorm2d: 2-20                 [8, 64, 32, 32]           128

#   │    └─ReLU: 2-21                        [8, 64, 32, 32]           --

#   │    └─Conv2d: 2-22                      [8, 64, 32, 32]           36,928

#   │    └─BatchNorm2d: 2-23                 [8, 64, 32, 32]           128

#   │    └─ReLU: 2-24                        [8, 64, 32, 32]           --

#   ├─Upsample: 1-8                          [8, 64, 64, 64]           --

#   ├─VGGBlock: 1-9                          [8, 32, 64, 64]           --

#   │    └─Conv2d: 2-25                      [8, 32, 64, 64]           27,680

#   │    └─BatchNorm2d: 2-26                 [8, 32, 64, 64]           64

#   │    └─ReLU: 2-27                        [8, 32, 64, 64]           --

#   │    └─Conv2d: 2-28                      [8, 32, 64, 64]           9,248

#   │    └─BatchNorm2d: 2-29                 [8, 32, 64, 64]           64

#   │    └─ReLU: 2-30                        [8, 32, 64, 64]           --

#   ├─Upsample: 1-10                         [8, 32, 128, 128]         --

#   ├─VGGBlock: 1-11                         [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-31                      [8, 16, 128, 128]         9,232

#   │    └─BatchNorm2d: 2-32                 [8, 16, 128, 128]         32

#   │    └─ReLU: 2-33                        [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-34                      [8, 16, 128, 128]         2,320

#   │    └─BatchNorm2d: 2-35                 [8, 16, 128, 128]         32

#   │    └─ReLU: 2-36                        [8, 16, 128, 128]         --

#   ├─MaxPool2d: 1-12                        [8, 64, 16, 16]           --

#   ├─VGGBlock: 1-13                         [8, 128, 16, 16]          --

#   │    └─Conv2d: 2-37                      [8, 128, 16, 16]          73,856

#   │    └─BatchNorm2d: 2-38                 [8, 128, 16, 16]          256

#   │    └─ReLU: 2-39                        [8, 128, 16, 16]          --

#   │    └─Conv2d: 2-40                      [8, 128, 16, 16]          147,584

#   │    └─BatchNorm2d: 2-41                 [8, 128, 16, 16]          256

#   │    └─ReLU: 2-42                        [8, 128, 16, 16]          --

#   ├─Upsample: 1-14                         [8, 128, 32, 32]          --

#   ├─VGGBlock: 1-15                         [8, 64, 32, 32]           --

#   │    └─Conv2d: 2-43                      [8, 64, 32, 32]           110,656

#   │    └─BatchNorm2d: 2-44                 [8, 64, 32, 32]           128

#   │    └─ReLU: 2-45                        [8, 64, 32, 32]           --

#   │    └─Conv2d: 2-46                      [8, 64, 32, 32]           36,928

#   │    └─BatchNorm2d: 2-47                 [8, 64, 32, 32]           128

#   │    └─ReLU: 2-48                        [8, 64, 32, 32]           --

#   ├─Upsample: 1-16                         [8, 64, 64, 64]           --

#   ├─VGGBlock: 1-17                         [8, 32, 64, 64]           --

#   │    └─Conv2d: 2-49                      [8, 32, 64, 64]           36,896

#   │    └─BatchNorm2d: 2-50                 [8, 32, 64, 64]           64

#   │    └─ReLU: 2-51                        [8, 32, 64, 64]           --

#   │    └─Conv2d: 2-52                      [8, 32, 64, 64]           9,248

#   │    └─BatchNorm2d: 2-53                 [8, 32, 64, 64]           64

#   │    └─ReLU: 2-54                        [8, 32, 64, 64]           --

#   ├─Upsample: 1-18                         [8, 32, 128, 128]         --

#   ├─VGGBlock: 1-19                         [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-55                      [8, 16, 128, 128]         11,536

#   │    └─BatchNorm2d: 2-56                 [8, 16, 128, 128]         32

#   │    └─ReLU: 2-57                        [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-58                      [8, 16, 128, 128]         2,320

#   │    └─BatchNorm2d: 2-59                 [8, 16, 128, 128]         32

#   │    └─ReLU: 2-60                        [8, 16, 128, 128]         --

#   ├─MaxPool2d: 1-20                        [8, 128, 8, 8]            --

#   ├─VGGBlock: 1-21                         [8, 256, 8, 8]            --

#   │    └─Conv2d: 2-61                      [8, 256, 8, 8]            295,168

#   │    └─BatchNorm2d: 2-62                 [8, 256, 8, 8]            512

#   │    └─ReLU: 2-63                        [8, 256, 8, 8]            --

#   │    └─Conv2d: 2-64                      [8, 256, 8, 8]            590,080

#   │    └─BatchNorm2d: 2-65                 [8, 256, 8, 8]            512

#   │    └─ReLU: 2-66                        [8, 256, 8, 8]            --

#   ├─Upsample: 1-22                         [8, 256, 16, 16]          --

#   ├─VGGBlock: 1-23                         [8, 128, 16, 16]          --

#   │    └─Conv2d: 2-67                      [8, 128, 16, 16]          442,496

#   │    └─BatchNorm2d: 2-68                 [8, 128, 16, 16]          256

#   │    └─ReLU: 2-69                        [8, 128, 16, 16]          --

#   │    └─Conv2d: 2-70                      [8, 128, 16, 16]          147,584

#   │    └─BatchNorm2d: 2-71                 [8, 128, 16, 16]          256

#   │    └─ReLU: 2-72                        [8, 128, 16, 16]          --

#   ├─Upsample: 1-24                         [8, 128, 32, 32]          --

#   ├─VGGBlock: 1-25                         [8, 64, 32, 32]           --

#   │    └─Conv2d: 2-73                      [8, 64, 32, 32]           147,520

#   │    └─BatchNorm2d: 2-74                 [8, 64, 32, 32]           128

#   │    └─ReLU: 2-75                        [8, 64, 32, 32]           --

#   │    └─Conv2d: 2-76                      [8, 64, 32, 32]           36,928

#   │    └─BatchNorm2d: 2-77                 [8, 64, 32, 32]           128

#   │    └─ReLU: 2-78                        [8, 64, 32, 32]           --

#   ├─Upsample: 1-26                         [8, 64, 64, 64]           --

#   ├─VGGBlock: 1-27                         [8, 32, 64, 64]           --

#   │    └─Conv2d: 2-79                      [8, 32, 64, 64]           46,112

#   │    └─BatchNorm2d: 2-80                 [8, 32, 64, 64]           64

#   │    └─ReLU: 2-81                        [8, 32, 64, 64]           --

#   │    └─Conv2d: 2-82                      [8, 32, 64, 64]           9,248

#   │    └─BatchNorm2d: 2-83                 [8, 32, 64, 64]           64

#   │    └─ReLU: 2-84                        [8, 32, 64, 64]           --

#   ├─Upsample: 1-28                         [8, 32, 128, 128]         --

#   ├─VGGBlock: 1-29                         [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-85                      [8, 16, 128, 128]         13,840

#   │    └─BatchNorm2d: 2-86                 [8, 16, 128, 128]         32

#   │    └─ReLU: 2-87                        [8, 16, 128, 128]         --

#   │    └─Conv2d: 2-88                      [8, 16, 128, 128]         2,320

#   │    └─BatchNorm2d: 2-89                 [8, 16, 128, 128]         32

#   │    └─ReLU: 2-90                        [8, 16, 128, 128]         --

#   ├─Conv2d: 1-30                           [8, 8, 128, 128]          136

#   ==========================================================================================

#   Total params: 2,293,912

#   Trainable params: 2,293,912

#   Non-trainable params: 0

#   Total mult-adds (Units.GIGABYTES): 17.32

#   ==========================================================================================

#   Input size (MB): 1.57

#   Forward/backward pass size (MB): 549.45

#   Params size (MB): 9.18

#   Estimated Total Size (MB): 560.20

#   ==========================================================================================

model_nested = NestedUNet(number_of_classes)
model_nested = model_nested.to(device)
nested_Unet_training_val_summary = train_model(model_nested, device, train_dataloader, val_dataloader,
                                        lr=3e-4, epochs=cfg.SINGLE_NETWORK_TRAINING_EPOCHS,with_dice=True, with_iou=True)

# Output:
#   Epoch 1/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 1/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 2/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 2/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 3/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 3/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 4/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 4/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 5/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 5/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 6/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 6/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 7/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 7/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 8/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 8/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 9/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 9/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 10/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 10/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 11/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 11/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 12/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 12/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 13/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 13/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 14/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 14/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]
#   Epoch 15/15 - Training Loss: 0:   0%|          | 0/372 [00:00<?, ?it/s]
#   Epoch 15/15 - Validation Loss: 0:   0%|          | 0/63 [00:00<?, ?it/s]

plot_training_progress(nested_Unet_training_val_summary)
# Output:
#   <Figure size 2000x600 with 3 Axes>

batch = next(iter(val_dataloader))
predictions = model_nested(batch[0].to(device))
show_inference(batch, predictions)
# Output:
#   <Figure size 1200x3200 with 24 Axes>

"""
##Attention Unet
"""

class ConvBlockAttention(nn.Module):
    """Simplified convolution block with a single Conv layer, BatchNorm, and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownScale(nn.Module):
    """Downscaling with maxpool then ConvBlock, transforming an input with (h, w, in_channels) to (h/2, w/2, out_channels)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlockAttention(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x

class AttentionBlock(nn.Module):
    """Simplified Attention Gate."""
    def __init__(self, f_g, f_l, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv_g = nn.Conv2d(f_g, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_x = nn.Conv2d(f_l, out_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        psi = self.sigmoid(self.psi(F.relu(self.conv_g(g) + self.conv_x(x))))
        return x * psi

class UpScale(nn.Module):
    """Upscaling with a single ConvBlock and an optional Attention Gate."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ConvBlockAttention(in_channels, out_channels)
        self.attention = AttentionBlock(f_g=in_channels // 2, f_l=out_channels, out_channels=out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.attention(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionUnetLite(nn.Module):
    """Lightweight Attention U-Net with reduced depth and complexity."""
    def __init__(self, n_channels, n_classes, start=32, bilinear=False):
        super(AttentionUnetLite, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBlockAttention(n_channels, start)
        self.down1 = DownScale(start, 2*start)
        self.down2 = DownScale(2*start, 4*start)

        factor = 2 if bilinear else 1
        self.down3 = DownScale(4*start, 8*start // factor)

        self.up1 = UpScale(8*start, 4*start // factor, bilinear)
        self.up2 = UpScale(4*start, 2*start, bilinear)
        self.up3 = UpScale(2*start, start, bilinear)
        self.outc = nn.Conv2d(start, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


summary(AttentionUnetLite(3, number_of_classes), input_data=eval_batch_data[0])
# Output:
#   ==========================================================================================

#   Layer (type:depth-idx)                   Output Shape              Param #

#   ==========================================================================================

#   AttentionUnetLite                        [8, 8, 128, 128]          --

#   ├─ConvBlockAttention: 1-1                [8, 32, 128, 128]         --

#   │    └─Conv2d: 2-1                       [8, 32, 128, 128]         864

#   │    └─BatchNorm2d: 2-2                  [8, 32, 128, 128]         64

#   │    └─ReLU: 2-3                         [8, 32, 128, 128]         --

#   ├─DownScale: 1-2                         [8, 64, 64, 64]           --

#   │    └─MaxPool2d: 2-4                    [8, 32, 64, 64]           --

#   │    └─ConvBlockAttention: 2-5           [8, 64, 64, 64]           --

#   │    │    └─Conv2d: 3-1                  [8, 64, 64, 64]           18,432

#   │    │    └─BatchNorm2d: 3-2             [8, 64, 64, 64]           128

#   │    │    └─ReLU: 3-3                    [8, 64, 64, 64]           --

#   ├─DownScale: 1-3                         [8, 128, 32, 32]          --

#   │    └─MaxPool2d: 2-6                    [8, 64, 32, 32]           --

#   │    └─ConvBlockAttention: 2-7           [8, 128, 32, 32]          --

#   │    │    └─Conv2d: 3-4                  [8, 128, 32, 32]          73,728

#   │    │    └─BatchNorm2d: 3-5             [8, 128, 32, 32]          256

#   │    │    └─ReLU: 3-6                    [8, 128, 32, 32]          --

#   ├─DownScale: 1-4                         [8, 256, 16, 16]          --

#   │    └─MaxPool2d: 2-8                    [8, 128, 16, 16]          --

#   │    └─ConvBlockAttention: 2-9           [8, 256, 16, 16]          --

#   │    │    └─Conv2d: 3-7                  [8, 256, 16, 16]          294,912

#   │    │    └─BatchNorm2d: 3-8             [8, 256, 16, 16]          512

#   │    │    └─ReLU: 3-9                    [8, 256, 16, 16]          --

#   ├─UpScale: 1-5                           [8, 128, 32, 32]          --

#   │    └─ConvTranspose2d: 2-10             [8, 128, 32, 32]          131,200

#   │    └─AttentionBlock: 2-11              [8, 128, 32, 32]          --

#   │    │    └─Conv2d: 3-10                 [8, 128, 32, 32]          16,512

#   │    │    └─Conv2d: 3-11                 [8, 128, 32, 32]          16,512

#   │    │    └─Conv2d: 3-12                 [8, 1, 32, 32]            129

#   │    │    └─Sigmoid: 3-13                [8, 1, 32, 32]            --

#   │    └─ConvBlockAttention: 2-12          [8, 128, 32, 32]          --

#   │    │    └─Conv2d: 3-14                 [8, 128, 32, 32]          294,912

#   │    │    └─BatchNorm2d: 3-15            [8, 128, 32, 32]          256

#   │    │    └─ReLU: 3-16                   [8, 128, 32, 32]          --

#   ├─UpScale: 1-6                           [8, 64, 64, 64]           --

#   │    └─ConvTranspose2d: 2-13             [8, 64, 64, 64]           32,832

#   │    └─AttentionBlock: 2-14              [8, 64, 64, 64]           --

#   │    │    └─Conv2d: 3-17                 [8, 64, 64, 64]           4,160

#   │    │    └─Conv2d: 3-18                 [8, 64, 64, 64]           4,160

#   │    │    └─Conv2d: 3-19                 [8, 1, 64, 64]            65

#   │    │    └─Sigmoid: 3-20                [8, 1, 64, 64]            --

#   │    └─ConvBlockAttention: 2-15          [8, 64, 64, 64]           --

#   │    │    └─Conv2d: 3-21                 [8, 64, 64, 64]           73,728

#   │    │    └─BatchNorm2d: 3-22            [8, 64, 64, 64]           128

#   │    │    └─ReLU: 3-23                   [8, 64, 64, 64]           --

#   ├─UpScale: 1-7                           [8, 32, 128, 128]         --

#   │    └─ConvTranspose2d: 2-16             [8, 32, 128, 128]         8,224

#   │    └─AttentionBlock: 2-17              [8, 32, 128, 128]         --

#   │    │    └─Conv2d: 3-24                 [8, 32, 128, 128]         1,056

#   │    │    └─Conv2d: 3-25                 [8, 32, 128, 128]         1,056

#   │    │    └─Conv2d: 3-26                 [8, 1, 128, 128]          33

#   │    │    └─Sigmoid: 3-27                [8, 1, 128, 128]          --

#   │    └─ConvBlockAttention: 2-18          [8, 32, 128, 128]         --

#   │    │    └─Conv2d: 3-28                 [8, 32, 128, 128]         18,432

#   │    │    └─BatchNorm2d: 3-29            [8, 32, 128, 128]         64

#   │    │    └─ReLU: 3-30                   [8, 32, 128, 128]         --

#   ├─Conv2d: 1-8                            [8, 8, 128, 128]          264

#   ==========================================================================================

#   Total params: 992,619

#   Trainable params: 992,619

#   Non-trainable params: 0

#   Total mult-adds (Units.GIGABYTES): 13.26

#   ==========================================================================================

#   Input size (MB): 1.57

#   Forward/backward pass size (MB): 429.20

#   Params size (MB): 3.97

#   Estimated Total Size (MB): 434.74

#   ==========================================================================================

model = AttentionUnetLite(3, number_of_classes)
model = model.to(device)
AttentionUnet_training_val_summary = train_model(model, device, train_dataloader, val_dataloader,
                                        lr=3e-4, epochs=cfg.SINGLE_NETWORK_TRAINING_EPOCHS,with_dice=True, with_iou=True)

plot_training_progress(AttentionUnet_training_val_summary)
# Output:
#   <Figure size 2000x600 with 3 Axes>

batch = next(iter(val_dataloader))
predictions = model(batch[0].to(device))
show_inference(batch, predictions)
# Output:
#   <Figure size 1200x3200 with 24 Axes>

