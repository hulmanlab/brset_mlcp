# %% [markdown]
# ### Setup Environment:

# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 6"

from src.get_dataset import get_dataset, split_data, plot_labels_distribution
from src.data_loader import BRSETDataset, process_labels
from src.RetFound import get_retfound
from src.FocalLoss import FocalLoss
from src.model import FoundationalCVModel, FoundationalCVModelWithClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from src.train import train
from src.test import test
import argparse 

# %% [markdown]
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set backbone and backbone_mode for the model.")
parser.add_argument('-b','--backbone', type=str, required=True, choices=['convnextv2_large','resnet200d','retfound','dinov2_large','visionfm'], help="Specify the backbone model (convnextv2_large','resnet50','retfound','dinov2_large','visionfm').")
parser.add_argument('-bm', '--backbone_mode', type=str, required=True, choices=['fine_tune', 'eval'], help="Specify the backbone mode ('fine_tune' or 'eval').")
args = parser.parse_args()

# Assign parsed arguments to variables
BACKBONE = args.backbone 
backbone_mode = args.backbone_mode 
print(f"Backbone: {BACKBONE}")
print(f"Backbone mode: {backbone_mode}")
# %%
# Constants:
DATASET = '/home/livieymli/brset_analysis/mBRSET'

LABELS_PATH = os.path.join(DATASET, 'data/labels_mbrset.csv')
LABELS_PATH_TRAIN = os.path.join(DATASET, 'data/train_mbrset_nooverlap.csv') 
# LABELS_PATH_VAL = os.path.join(DATASET, 'data/val_mbrset_nooverlap.csv')
# LABELS_PATH_TEST = os.path.join(DATASET, 'data/test_mbrset_nooverlap.csv')
LABELS_PATH_TEST = LABELS_PATH
IMAGES = os.path.join(DATASET, 'data/images/')
DOWNLOAD = False
SHAPE = (224, 224)
LABEL = 'final_icdr'
IMAGE_COL = 'file'
TEST_SIZE = 0.3
GROUP_STRATIFY = False
UNDERSAMPLE = False

"""
Dataset Mean and Std:
NORM_MEAN = [0.5896205017400412, 0.29888971649817453, 0.1107679405196557]
NORM_STD = [0.28544273712830986, 0.15905456049750208, 0.07012281660980953]

ImageNet Mean and Std:
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
"""

NORM_MEAN = None # [0.485, 0.456, 0.406]
NORM_STD = None # [0.229, 0.224, 0.225]

# BACKBONE = 'convnextv2_large'
# BACKBONE = 'resnet200d'
# BACKBONE = 'retfound'
# BACKBONE = 'dinov2_large'
# BACKBONE = 'visionfm'

if BACKBONE == 'retfound':
    weights = '/home/livieymli/brset_analysis/BRSET/src/Weights/RETFound_cfp_weights.pth'

elif BACKBONE == 'visionfm':
    weights = {
        'arch' : 'vit_base',
        'image_size' : 224,
        'patch_size' : 16,
        'weights' : '/home/livieymli/brset_analysis/BRSET/src/Weights/VFM_Fundus_weights.pth'
    }   
else:
    weights = None
MODE = 'fine_tune'
# backbone_mode = 'fine_tune' # 'fine_tune' or 'eval'
# backbone_mode = 'eval'

HIDDEN = [128]
num_classes = 3

BATCH_SIZE = 16
NUM_WORKERS = 4

LOSS = None #'focal_loss'
OPTIMIZER = 'adam'

# Define your hyperparameters
num_epochs = 50
learning_rate = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using", torch.cuda.device_count(), "GPUs!")

# %% [markdown]
# #### Read csv file:

# %%
# df = get_dataset(LABELS_PATH, download=DOWNLOAD, info=False)
df_train = get_dataset(LABELS_PATH_TRAIN, download=DOWNLOAD, info=False, name='mBRSET')
df_test = get_dataset(LABELS_PATH_TEST, download=DOWNLOAD, info=False, name='mBRSET')
# df_val = get_dataset(LABELS_PATH_VAL, download=DOWNLOAD, info=False, name='mBRSET')
df_test.head()

# %%
# Convert into 3 classes:

# Normal = 0; Non-proliferative = 1, 2, 3; Proliferative = 4
# Map values to categories
df_train[LABEL] = df_train[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
df_test[LABEL] = df_test[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
# df_val[LABEL] = df_val[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))


# %% [markdown]
# ### Dataloaders

# %%
# Train the one hot encoder on the train set and get the labels for the test and validation sets:
train_labels, mlb, train_columns = process_labels(df_train, col=LABEL)

# %%
# Define the target image shape

# train_transforms = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomCrop(SHAPE),
#     transforms.ToTensor(),
#     transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
#     transforms.RandomRotation(50),  # Randomly rotate the image by up to 10 degrees
# ])

# if NORM_MEAN is not None and NORM_STD is not None:
#     train_transforms.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

test_transform = transforms.Compose([
    transforms.Resize(SHAPE),
    transforms.ToTensor(),
])

if NORM_MEAN is not None and NORM_STD is not None:
    test_transform.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

# %%
# Create the custom dataset
# train_dataset = BRSETDataset(
#     df_train, 
#     IMAGE_COL, 
#     IMAGES, 
#     LABEL, 
#     mlb, 
#     train_columns, 
#     transform=train_transforms
# )

test_dataset = BRSETDataset(
    df_test, 
    IMAGE_COL, 
    IMAGES, 
    LABEL, 
    mlb, 
    train_columns, 
    transform=test_transform
)

# val_dataset = BRSETDataset(
#     df_val, 
#     IMAGE_COL, 
#     IMAGES, 
#     LABEL, 
#     mlb, 
#     train_columns, 
#     transform=test_transform
# )

# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
# val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# %%
# Print 6 samples with their labels
# Iterate through the DataLoader and plot the images with labels

for batch in test_dataloader:

    images, labels = batch['image'], batch['labels']

    for i in range(len(images)):
        if i == 6:
            break
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Permute to (H, W, C) from (C, H, W)
        plt.title(f"Label: {np.argmax(labels[i])}")
        plt.axis('off')
    plt.show()
    break

# %% [markdown]
# ### Model

# %%
# Create a DataLoader to generate embeddings
backbone_model = FoundationalCVModel(backbone=BACKBONE, mode=MODE, weights=weights)
model = FoundationalCVModelWithClassifier(backbone_model, hidden=HIDDEN, num_classes=num_classes, mode=MODE, backbone_mode=backbone_mode)
model.to(device)

# Use DataParallel to parallelize the model across multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, [0,1])


# %% [markdown]
# ### Test

# %%
path = f'/home/livieymli/brset_analysis/BRSET/models/FT_{BACKBONE}_{backbone_mode}_3class_DR_ICDR_best.pth'
net = torch.load(path, map_location=torch.device(device))
if device.type == 'cpu':
    net = {k.replace('module.', ''): v for k, v in net.items()}
    # net = {k.replace('backbone.', ''): v for k, v in net.items()}
model.load_state_dict(net, strict=False)
#%%
test(model, test_dataloader, saliency=True, device=device, save_prob=True, prob_name=f'temp_{BACKBONE}_{backbone_mode}_3class_mBRSET_EXEVAL', save=True)

# %% [markdown]
# ### Image quality assessment

# %%
# Good quality images
adequate_df = df_test[df_test['final_quality'] == 'yes']

# Bad quality images
inadequate_df = df_test[df_test['final_quality'] == 'no']

adequate_dataset = BRSETDataset(
    adequate_df, 
    IMAGE_COL, 
    IMAGES, 
    LABEL, 
    mlb, 
    train_columns, 
    transform=test_transform
)

inadequate_dataset = BRSETDataset(
    inadequate_df, 
    IMAGE_COL, 
    IMAGES, 
    LABEL, 
    mlb, 
    train_columns, 
    transform=test_transform
)

adequate_dataloader = DataLoader(adequate_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
inadequate_dataloader = DataLoader(inadequate_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# %% [markdown]
# #### Adequate image quality

# %%
test(model, adequate_dataloader, saliency=True, device=device, save=True)

# %% [markdown]
# #### Inadequate image quality

# %%
test(model, inadequate_dataloader, saliency=True, device=device)

# %%



