# %% [markdown]
# ### Setup Environment:

# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"  # Specify the GPUs to use
import torch, gc
gc.collect()
torch.cuda.empty_cache()

from src.get_dataset import get_dataset, split_data, plot_labels_distribution
from src.data_loader import BRSETDataset, process_labels, BRSET_mBRSETDataset
from src.RetFound import get_retfound
from src.FocalLoss import FocalLoss
from src.model import FoundationalCVModel, FoundationalCVModelWithClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from src.train import train
from src.test import test
import argparse 

# %% Parse command-line arguments
parser = argparse.ArgumentParser(description="Set backbone and backbone_mode for the model.")
parser.add_argument('-b','--backbone', type=str, required=True, choices=['convnextv2_large','resnet200d','retfound','dinov2_large','visionfm'], help="Specify the backbone model (convnextv2_large','resnet50','retfound','dinov2_large','visionfm').")
parser.add_argument('-bm', '--backbone_mode', type=str, required=True, choices=['fine_tune', 'eval'], help="Specify the backbone mode ('fine_tune' or 'eval').")
parser.add_argument('-r', '--reproduce', type=bool, default=False, help="Specify if you want to reproduce the results from the article (True or False).")
parser.add_argument('-t', '--test', type=bool, default=False, help="Specify if you want to test the model (True or False).")
args = parser.parse_args()

# Assign parsed arguments to variables
BACKBONE = args.backbone
backbone_mode = args.backbone_mode
reproduce = False
TEST = args.test
#%%

print(f"Using backbone: {BACKBONE}, backbone_mode: {backbone_mode}, reproduce: {reproduce}, test={TEST}")
# %%
# Constants:
DATASET = os.path.dirname(os.path.realpath(__name__))


# LABELS_PATH = os.path.join(DATASET, 'data/labels_brset.csv')
LABELS_PATH_TRAIN = os.path.join(DATASET, 'data/train_70&70.csv') 
LABELS_PATH_VAL = os.path.join(DATASET, 'data/val_70&70.csv')
LABELS_PATH_TEST = os.path.join(DATASET, 'data/test_70&70.csv')

LABELS_PATH_TEST_BRSET = os.path.join(DATASET, 'data/test_brset_nooverlap.csv')
LABELS_PATH_TEST_mBRSET = os.path.join(DATASET, 'data/test_mbrset_nooverlap.csv')

DOWNLOAD = False
SHAPE = (224, 224)
LABEL = 'label'
IMAGE_COL = 'file_path'
TEST_SIZE = 0.3
UNDERSAMPLE = False

IMAGES_B = os.path.join(DATASET, 'data/fundus_photos/')
IMAGES_m = os.path.join(DATASET, 'data/images/')

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
# backbone_mode = 'fine_tune' # 'fine_tune' or 'eval'
# backbone_mode = 'eval'

HIDDEN = [128]
num_classes = 3

BATCH_SIZE = 16
NUM_WORKERS = 8

LOSS = None #'focal_loss'
OPTIMIZER = 'adam'

# Define your hyperparameters
num_epochs = 50
learning_rate = 1e-5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using", torch.cuda.device_count(), "GPUs!")

# %% [markdown]
# #### Read csv file:

# df = get_dataset(LABELS_PATH, download=DOWNLOAD, info=False)
df_train = get_dataset(LABELS_PATH_TRAIN, download=DOWNLOAD, info=False, name='70&70')
df_test = get_dataset(LABELS_PATH_TEST, download=DOWNLOAD, info=False, name='70&70')
df_val = get_dataset(LABELS_PATH_VAL, download=DOWNLOAD, info=False, name='70&70')

df_test_B = get_dataset(LABELS_PATH_TEST_BRSET, download=DOWNLOAD, info=False)
df_test_m = get_dataset(LABELS_PATH_TEST_mBRSET, download=DOWNLOAD, info=False, name='mBRSET')
# %%
# Convert into 3 classes:

# Normal = 0; Non-proliferative = 1, 2, 3; Proliferative = 4
# Map values to categories
df_train[LABEL] = df_train[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
df_test[LABEL] = df_test[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
df_val[LABEL] = df_val[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))

df_train['camera'] = df_train['camera'].apply(lambda x: 0 if x == 'Canon CR' else 1)
df_val['camera'] = df_val['camera'].apply(lambda x: 0 if x == 'Canon CR' else 1)
df_test['camera'] = df_test['camera'].apply(lambda x: 0 if x == 'Canon CR' else 1)
df_test_B['camera'] = df_test_B['camera'].apply(lambda x: 0 if x == 'Canon CR' else 1)

df_test_B['DR_ICDR'] = df_test_B['DR_ICDR'].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
df_test_m['final_icdr'] = df_test_m['final_icdr'].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
# %% [markdown]
# ### Dataloaders

# Train the one hot encoder on the train set and get the labels for the test and validation sets:
train_labels, mlb, train_columns = process_labels(df_train, col=LABEL)
test_labels_B, mlb_B, test_columns_B = process_labels(df_test_B, col='DR_ICDR')
test_labels_m, mlb_m, test_columns_m = process_labels(df_test_m, col='final_icdr')
# %%
# Define the target image shape

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
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
train_dataset = BRSET_mBRSETDataset(df_train, IMAGE_COL, IMAGES_B, LABEL, mlb, train_columns, 
                             transform=train_transforms)

val_dataset = BRSET_mBRSETDataset(df_val, IMAGE_COL, IMAGES_B, LABEL, mlb, train_columns, 
                           transform=test_transform
)

test_dataset = BRSET_mBRSETDataset(df_test, IMAGE_COL, IMAGES_B, LABEL, mlb, train_columns,
                           transform=test_transform
)

test_dataset_B = BRSETDataset(
    df_test_B, 
    'image_id', 
    IMAGES_B, 
    'DR_ICDR', 
    mlb_B, 
    test_columns_B, 
    transform=test_transform
)

test_dataset_m = BRSETDataset(
    df_test_m, 
    'file', 
    IMAGES_m, 
    'final_icdr', 
    mlb_m, 
    test_columns_m, 
    transform=test_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


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
        plt.title(f"Label: {np.argmax(labels[i])}")
        plt.axis('off')
    plt.show()
    break


# %% [markdown]
# ### Model

# Create a DataLoader to generate embeddings
# model = get_retfound(weights='/home/opc/FoundationalRetina/Weights/RETFound_cfp_weights.pth', num_classes=3)
# Create a DataLoader to generate embeddings
backbone_model = FoundationalCVModel(backbone=BACKBONE, mode=MODE, weights=weights)
model = FoundationalCVModelWithClassifier(backbone_model, hidden=HIDDEN, num_classes=num_classes, mode=MODE, backbone_mode=backbone_mode)
model.to(device)

# Use DataParallel to parallelize the model across multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, [0,1])  # Specify the GPUs to use

model.to(device)
# %% [markdown]
# ### Training:

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
if TEST:
    print("Testing the model...")
else:
    print("Training the model...")
    model = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs=num_epochs, save=True, device=device, ppath=DATASET, backbone=f'70+70_{BACKBONE}_{backbone_mode}_3class{"_reproduce" if reproduce == True else ""}_{LABEL}')

# %% [markdown]
# ### Load Trained Model 
path = os.path.join(DATASET, f'output/models/FT_70+70_{BACKBONE}_{backbone_mode}_3class_{LABEL}_best.pth')
net = torch.load(path, map_location=torch.device(device))
if device.type == 'cpu':
    net = {k.replace('module.', ''): v for k, v in net.items()}
    # net = {k.replace('backbone.', ''): v for k, v in net.items()}
model.load_state_dict(net, strict=False)
#%% Test
test(model, test_dataloader, saliency=True, device=device, save_prob=True, prob_name=f'70+70_{BACKBONE}_{backbone_mode}_3class{"_reproduce" if reproduce == True else ""}')

test_dataloader_B = DataLoader(test_dataset_B, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_dataloader_m = DataLoader(test_dataset_m, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

test(model, test_dataloader_B, saliency=True, device=device, save_prob=True, prob_name=f'70+70_BRSET_{BACKBONE}_{backbone_mode}_3class{"_reproduce" if reproduce == True else ""}')
test(model, test_dataloader_m, saliency=True, device=device, save_prob=True, prob_name=f'70+70_mBRSET_{BACKBONE}_{backbone_mode}_3class{"_reproduce" if reproduce == True else ""}')


