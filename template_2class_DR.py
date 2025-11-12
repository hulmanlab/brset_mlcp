# %% [markdown]
# ### Setup Environment:
# %%
import os

from src.get_dataset import get_dataset, split_data, plot_labels_distribution
from src.data_loader import BRSETDataset, process_labels
from src.model import FoundationalCVModel, FoundationalCVModelWithClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# loss function and optimizer
from src.FocalLoss import BinaryFocalLoss, FocalLoss

# train and test functions
from src.train import train
from src.test import test
import argparse 

# %% [markdown]
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set backbone and backbone_mode for the model.")
parser.add_argument('-b','--backbone', type=str, required=True, choices=['convnextv2_large','resnet50','retfound','dinov2_large','visionfm'], help="Specify the backbone model (convnextv2_large','resnet50','retfound','dinov2_large','visionfm').")
parser.add_argument('-bm', '--backbone_mode', type=str, required=True, choices=['fine_tune', 'eval'], help="Specify the backbone mode ('fine_tune' or 'eval').")
parser.add_argument('-r', '--reproduce', type=bool, default=False, help="Specify if you want to reproduce the results from the article (True or False).")
args = parser.parse_args()

# Assign parsed arguments to variables
BACKBONE = args.backbone 
backbone_mode = args.backbone_mode
reproduce = args.reproduce 
print(f'Backbone: {BACKBONE}')
print(f'Backbone mode: {backbone_mode}')
if reproduce:
    print(f'Reproduce: {reproduce}')
# %%
# Constants:
DATASET = os.path.dirname(os.path.realpath(__name__))
DOWNLOAD = False
SHAPE = (224, 224)
IMAGES = os.path.join(DATASET, 'data/fundus_photos/')
LABEL = 'DR_ICDR'
TEST_SIZE = 0.3
UNDERSAMPLE = False

LABELS_PATH = os.path.join(DATASET, 'data/labels_brset.csv')
LABELS_PATH_TRAIN = os.path.join(DATASET, 'data/train_brset_nooverlap.csv') 
LABELS_PATH_VAL = os.path.join(DATASET, 'data/val_brset_nooverlap.csv')
LABELS_PATH_TEST = os.path.join(DATASET, 'data/test_brset_nooverlap.csv')

if reproduce:
    LABELS_PATH_TRAIN = os.path.join(DATASET, 'data/train_brset.csv') 
    LABELS_PATH_VAL = os.path.join(DATASET, 'data/val_brset.csv')
    LABELS_PATH_TEST = os.path.join(DATASET, 'data/test_brset.csv') 
    backbone_mode = 'fine_tune' 

IMAGE_COL = 'image_id'


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
# BACKBONE = 'resnet50'
# BACKBONE = 'retfound'
# BACKBONE = 'dinov2_large'
# BACKBONE = 'visionfm'

if BACKBONE == 'retfound':
    weights = os.path.join(DATASET, 'src/Weights/RETFound_cfp_weights.pth')
elif BACKBONE == 'visionfm':
    weights = {
        'arch' : 'vit_base',
        'image_size' : 224,
        'patch_size' : 16,
        'weights' : os.path.join(DATASET, 'src/Weights/VFM_Fundus_weights.pth')
    }   
else:
    weights = None

MODE = 'fine_tune'
# backbone_mode = 'fine_tune' 
# backbone_mode = 'eval'

HIDDEN = [128]
num_classes = 2

BATCH_SIZE = 16
NUM_WORKERS = 4

LOSS = None #'focal_loss'
OPTIMIZER = 'adam'

# Define your hyperparameters
num_epochs = 50
learning_rate = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set the specific GPU to use

print("Using", torch.cuda.device_count(), "GPUs!")


# %% [markdown]

# df = get_dataset(LABELS_PATH, download=DOWNLOAD, info=False)
# df.head()
df_train = get_dataset(LABELS_PATH_TRAIN, download=DOWNLOAD, info=False)
df_val = get_dataset(LABELS_PATH_VAL, download=DOWNLOAD, info=False)
df_test = get_dataset(LABELS_PATH_TEST, download=DOWNLOAD, info=False)

# %%
# Convert into 2 classes:

# Normal = 0; Non-proliferative = 1, 2, 3; Proliferative = 4
# Map values to categories
# df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x == 0 else 'Diabetic Retinopathy')
df_train[LABEL] = df_train[LABEL].apply(lambda x: 'Normal' if x == 0 else 'Diabetic Retinopathy')
df_val[LABEL] = df_val[LABEL].apply(lambda x: 'Normal' if x == 0 else 'Diabetic Retinopathy')
df_test[LABEL] = df_test[LABEL].apply(lambda x: 'Normal' if x == 0 else 'Diabetic Retinopathy')

# %%
def stratify_group(df, X_col, y_col, group_col, n_splits=1, shuffle = False, random_state=42):
    if shuffle == False:
        random_state = None
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    X = df[X_col].values
    y = df[y_col].values
    groups = df[group_col].values
    for train_index, test_index in skf.split(X, y, groups):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        break

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_labels_distribution(train_data, y_col, title='Train Label Distribution')
    plt.subplot(1, 2, 2)
    plot_labels_distribution(test_data, y_col, title='Test Label Distribution')
    plt.show()
    return train_data, test_data
        

# %%
# Split dataset into train, test and validation:
# if GROUP_STRATIFY == True:
#     df_train, df_test = stratify_group(df, 'image_id', LABEL, 'patient_id', n_splits=3, shuffle=True)
#     print('Getting validation set...')
#     df_test, df_val = stratify_group(df_test, 'image_id', LABEL, 'patient_id', n_splits=5, shuffle=False)
    
# else:
#     df_train, df_test = split_data(df, LABEL, TEST_SIZE, undersample=False)
#     print('Getting validation set...')
#     df_test, df_val = split_data(df_test, LABEL, 0.20)
    
    

# %% [markdown]
# ### Dataloaders

# %%
# Train the one hot encoder on the train set and get the labels for the test and validation sets:
train_labels, mlb, train_columns = process_labels(df_train, col=LABEL)

# %%
# Define the target image shape
SHAPE = (224, 224)  # Adjust to your desired image size

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(SHAPE),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(50),  # Randomly rotate the image by up to 10 degrees
])

if NORM_MEAN is not None and NORM_STD is not None:
    train_transforms.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

test_transform = transforms.Compose([
    transforms.Resize(SHAPE),
    transforms.ToTensor(),
])

if NORM_MEAN is not None and NORM_STD is not None:
    test_transform.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))


# %%
# Create the custom dataset
train_dataset = BRSETDataset(
    df_train, 
    IMAGE_COL, 
    IMAGES, 
    LABEL, 
    mlb, 
    train_columns, 
    transform=train_transforms
)

test_dataset = BRSETDataset(
    df_test, 
    IMAGE_COL, 
    IMAGES, 
    LABEL, 
    mlb, 
    train_columns, 
    transform=test_transform
)

val_dataset = BRSETDataset(
    df_val, 
    IMAGE_COL, 
    IMAGES, 
    LABEL, 
    mlb, 
    train_columns, 
    transform=test_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# %%
# Print 6 samples with their labels
# Iterate through the DataLoader and plot the images with labels
for batch in train_dataloader:
    images, labels = batch['image'], batch['labels']

    for i in range(len(images)):
        if i == 6:
            break
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Permute to (H, W, C) from (C, H, W)
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()
    break

# %% [markdown]
# ### Model

# %%
# Create the model
backbone_model = FoundationalCVModel(backbone=BACKBONE, mode=MODE, weights=weights)
model = FoundationalCVModelWithClassifier(backbone_model, hidden=HIDDEN, num_classes=num_classes, mode=MODE, backbone_mode=backbone_mode)
model.to(device)

# Use DataParallel to parallelize the model across multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, [0,1])

# %% [markdown]
# ### Training:

# %%
if LOSS == 'focal_loss':
    class_distribution = train_dataloader.dataset.labels.sum(axis=0)
    print(f'Class distribution: {class_distribution}')
    class_dis = np.array(class_distribution)
    class_weights =1-class_dis/np.sum(class_dis)
    weights = torch.tensor(class_weights).to(device)
    #criterion = FocalLoss()  # Focal Loss
    criterion = FocalLoss(gamma=2, alpha=weights)
else:
    # Assuming train_loader.dataset.labels is a one-hot representation
    class_indices = np.argmax(train_dataloader.dataset.labels, axis=1)

    # Compute class weights using class indices
    class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    #criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy Loss

if OPTIMIZER == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif OPTIMIZER == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4)
# %%
model = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs=num_epochs, save=True, device=device, ppath=DATASET, backbone=f'{BACKBONE}_{backbone_mode}_binary{"_reproduce" if reproduce == True else ""}_{LABEL}')

# %% [markdown]
# ### Test

# %%
path = os.path.join(DATASET, f'output/models/FT_{BACKBONE}_{backbone_mode}_binary_{LABEL}_best.pth')
net = torch.load(path, map_location=torch.device(device))
if device.type == 'cpu':
    net = {k.replace('module.', ''): v for k, v in net.items()}
    # net = {k.replace('backbone.', ''): v for k, v in net.items()}
model.load_state_dict(net, strict=False)


# %%
test(model, test_dataloader, saliency=True, device=device, save_prob=True,prob_name=f'{BACKBONE}_{backbone_mode}_binary')

# %% [markdown]
# ### Image quality assessment

# %%
# Good quality images
adequate_df = df_test[df_test['quality'] == 'Adequate']

# Bad quality images
inadequate_df = df_test[df_test['quality'] == 'Inadequate']

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



