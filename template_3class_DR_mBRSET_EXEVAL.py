# %% [markdown]
# ### Setup Environment:
# %%
import os
import torch
from datetime import timedelta
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def setup_ddp():
    if torch.cuda.is_available() and "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=15))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, local_rank, device

    # CPU or single-process GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, device

from src.get_dataset import get_dataset
from src.data_loader import BRSETDataset, process_labels
from src.model import FoundationalCVModel, FoundationalCVModelWithClassifier
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

# test functions
from src.test import test
import argparse 

# %% Constants:
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set backbone and backbone_mode for the model.")
parser.add_argument('-b','--backbone', type=str, required=True, choices=['retfound_d2_s','retfound_d2_m','dinov3_large','dinov2_large','visionfm', 'retfound'], help="Specify the backbone model (retfound_d2_s, retfound_d2_m, dinov3_large, dinov2_large, visionfm).")
parser.add_argument('-bm', '--backbone_mode', type=str, required=True, choices=['fine_tune', 'eval'], help="Specify the backbone mode ('fine_tune' or 'eval').")
args = parser.parse_args()

# Assign parsed arguments to variables
BACKBONE = args.backbone 
backbone_mode = args.backbone_mode 
print(f"Backbone: {BACKBONE}")
print(f"Backbone mode: {backbone_mode}")
# %%
# Constants:
DATASET = os.path.dirname(os.path.realpath(__name__))

LABELS_PATH = os.path.join(DATASET, 'data/labels_mbrset.csv')
LABELS_PATH_TEST = LABELS_PATH
IMAGES = os.path.join(DATASET, 'data/fundus_photos/')
DOWNLOAD = False
SHAPE = (224, 224)
LABEL = 'final_icdr'
IMAGE_COL = 'file'


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



if BACKBONE == 'retfound':
    weights = os.path.join(DATASET, 'src/Weights/RETFound_cfp_weights.pth')
elif BACKBONE == 'retfound_d2_s':
    weights = os.path.join(DATASET, 'src/Weights/RETFound_dinov2_shanghai.pth')
elif BACKBONE == 'retfound_d2_m':
    weights = os.path.join(DATASET, 'src/Weights/RETFound_dinov2_meh.pth')
elif BACKBONE == 'dinov3_large':
    weights = os.path.join(DATASET, 'src/Weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
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
num_classes = 3

BATCH_SIZE = 16
NUM_WORKERS = 4

LOSS = None #'focal_loss'
OPTIMIZER = 'adam'

# Define your hyperparameters
num_epochs = 50
learning_rate = 1e-5


ddp, local_rank, device = setup_ddp()
if ddp:
    is_main_process = (torch.distributed.get_rank() == 0)
else:
    is_main_process = True


print("Using", torch.cuda.device_count(), "GPUs!")


# %% [markdown]

# Get the dataset
# df.head()
df_test = get_dataset(LABELS_PATH_TEST, download=DOWNLOAD, info=False, name='mBRSET')

# %%
# Convert into 3 classes:

# Normal = 0; Non-proliferative = 1, 2, 3; Proliferative = 4
# Map values to categories
df_test[LABEL] = df_test[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))


# %% [markdown]
# ### Dataloaders

# %%
# Train the one hot encoder on the train set and get the labels for the test and validation sets:
test_labels, mlb, test_columns = process_labels(df_test, col=LABEL)

# %%
# Define the target image shape

test_transform = transforms.Compose([
    transforms.Resize(SHAPE),
    transforms.ToTensor(),
])

if NORM_MEAN is not None and NORM_STD is not None:
    test_transform.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

# %%
# Create the custom dataset

test_dataset = BRSETDataset(
    df_test, 
    IMAGE_COL, 
    IMAGES, 
    LABEL, 
    mlb, 
    test_columns, 
    transform=test_transform
)


test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory = (device.type == "cuda"))

# %%
# Print 6 samples with their labels
# Iterate through the DataLoader and plot the images with labels
# if not ddp:
#     for batch in train_dataloader:
#         images, labels = batch['image'], batch['labels']
#         for i in range(len(images)):
#             if i == 6:
#                 break
#             plt.subplot(2, 3, i + 1)
#             plt.imshow(images[i].permute(1, 2, 0))  # Permute to (H, W, C) from (C, H, W)
#             plt.title(f"Label: {labels[i]}")
#             plt.axis('off')
#         plt.show()
#         break

# %% [markdown]
# ### Model

# %%
# Create the model
backbone_model = FoundationalCVModel(backbone=BACKBONE, mode=MODE, weights=weights)
model = FoundationalCVModelWithClassifier(backbone_model, hidden=HIDDEN, num_classes=num_classes, mode=MODE, backbone_mode=backbone_mode)
model.to(device)

print(
    f"[rank {torch.distributed.get_rank() if ddp else 0}] "
    f"num params = {sum(p.numel() for p in model.parameters())}"
)

if ddp:
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True,
        broadcast_buffers=False,
        find_unused_parameters=False
    )
# %% [markdown]
# ### Test
 
# %%
path = os.path.join(DATASET, f'output/models/FT_{BACKBONE}_{backbone_mode}_3class_DR_ICDR_best.pth')
net = torch.load(path, map_location=device)

# Handle DDP / non-DDP key differences
if ddp:
    model.module.load_state_dict(net, strict=False)
else:
    model.load_state_dict(net, strict=False)


if is_main_process:
    test(model, test_dataloader, saliency=True, device=device, save_prob=True, prob_name=f'EX_{BACKBONE}_{backbone_mode}_3class')

#%% frees distributed resources
if ddp:
    torch.distributed.destroy_process_group()
