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

from src.data_loader import BRSETDataset, process_labels
from src.model import FoundationalCVModel
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

import numpy as np
import pandas as pd


# train and test functions
import argparse 

# %% [markdown]
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set backbone and backbone_mode for the model.")
parser.add_argument('-b','--backbone', type=str, required=True, choices=['retfound_d2_s','retfound_d2_m','dinov3_large','dinov2_large','visionfm', 'retfound'], help="Specify the backbone model (retfound_d2_s, retfound_d2_m, dinov3_large, dinov2_large, visionfm).")
parser.add_argument('-c','--camera_type', type=str, required=True, choices = ['Canon', 'Nikon'], help='Specify the camera type ("Canon" or "Nikon")')
parser.add_argument('-bm', '--backbone_mode', type=str, required=True, choices=['fine_tune', 'eval'], help="Specify the backbone mode ('fine_tune' or 'eval').")
parser.add_argument('-tp', '--timepoint', type=str, required=True, choices=['b', 'a'], help="Specify the timepoint ('b' for before_finetune or 'a' for after_finetune).")
args = parser.parse_args()

# Assign parsed arguments to variables
BACKBONE = args.backbone 
backbone_mode = args.backbone_mode
DATA_SOURCE = args.camera_type
tp = args.timepoint


# %% 
# BACKBONE = 'retfound'
# backbone_mode = 'eval'
# DATA_SOURCE = 'Canon'
# tp = 'a'

#%%
print(f"Using backbone: {BACKBONE}, backbone_mode: {backbone_mode}, dataset: {DATA_SOURCE}, time point: {'before fine-tuning' if tp == 'b' else 'after fine-tuning'}")

# Constants:
DATASET = os.path.dirname(os.path.realpath(__name__))
IMAGES = os.path.join(DATASET, 'data/fundus_photos/')
IMAGE_COL = 'image_id'
if DATA_SOURCE == 'Canon':
    LABELS_PATH = os.path.join(DATASET, 'data/labels_brset_Canon.csv')
elif DATA_SOURCE == 'Nikon':
    LABELS_PATH = os.path.join(DATASET, 'data/labels_brset_NIKON.csv')


DOWNLOAD = False
SHAPE = (224, 224)
LABEL = 'DR_ICDR'
UNDERSAMPLE = False


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

HIDDEN = [128]

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
# #### Read csv file:

df = pd.read_csv(LABELS_PATH)
# Convert into 3 classes:

# Normal = 0; Non-proliferative = 1, 2, 3; Proliferative = 4
# Map values to categories
df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))


# %% Train the one hot encoder on the train set and get the labels for the test and validation sets:
train_labels, mlb, train_columns = process_labels(df, col=LABEL)

# %% Create the test transforms

test_transform = transforms.Compose([
    transforms.Resize(SHAPE),
    transforms.ToTensor(),
])

if NORM_MEAN is not None and NORM_STD is not None:
    test_transform.transforms.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

# %%

test_dataset = BRSETDataset(
    df, 
    IMAGE_COL, 
    IMAGES, 
    LABEL, 
    mlb, 
    train_columns, 
    transform=test_transform
)


test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# %%
def generate_embeddings(batch, batch_number, model):
    """
    Generate image embeddings for a batch of images using the specified model.

    Parameters:
    - batch (tuple): A batch of images where the first element is a list of image names, and the second element is a tensor of images.
    - batch_number (int): The batch number for tracking progress.
    - model (torch.nn.Module): The model used to generate image embeddings.

    Returns:
    tuple: A tuple containing a list of image names and their corresponding embeddings.

    Example Usage:
    ```python
    img_names, embeddings = generate_embeddings(batch, batch_number, model)
    ```

    Note:
    - This function processes a batch of images and generates embeddings for each image.
    - It is typically used in a data loading pipeline to generate embeddings for a dataset.
    """
    image_names, images = batch['image_id'], batch['image']

    with torch.no_grad():
        features = model(images)

    if batch_number % 10 == 0:
        print(f"Processed batch number: {batch_number}")

    return image_names, features
# %% [markdown]
# ### Model

backbone_model = FoundationalCVModel(backbone=BACKBONE, mode=backbone_mode, weights=weights)
backbone_model.to(device)

# Use DataParallel to parallelize the model across multiple GPUs
print(
    f"[rank {torch.distributed.get_rank() if ddp else 0}] "
    f"num params = {sum(p.numel() for p in backbone_model.parameters())}"
)

if ddp:
    backbone_model = DDP(
        backbone_model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True,
        broadcast_buffers=False,
        find_unused_parameters=False
    )


if tp == 'a':
    path = os.path.join(DATASET, f'output/models/FT_{BACKBONE}_{backbone_mode}_3class_{LABEL}_best.pth')
    # All ranks load the SAME checkpoint
    net = torch.load(path, map_location=device)

    # Handle DDP / non-DDP key differences
    if ddp:
        backbone_model.module.load_state_dict(net, strict=False)
    else:
        backbone_model.load_state_dict(net, strict=False)
    
# %% [markdown]
# ### Generate embeddings
img_names = []
embeddings_list = np.empty((0, 1024))
for batch_number, batch in enumerate(test_dataloader, start=1):
    img_names_aux, features_aux = generate_embeddings(batch, batch_number, backbone_model)
    # Convert features to numpy array once per batch
    features_np = features_aux.cpu()
    # Save image ids and features as a dictionary to a .pt file
    # Extend embeddings_list with tuples of (image_id, embedding)
    img_names.extend(img_names_aux)
    embeddings_list = np.concatenate([embeddings_list, features_np], axis=0)
    if batch_number % 20 == 0:
        print(f"Embeddings list shape after batch {batch_number}: {embeddings_list.shape}")
# Create DataFrame directly from list of tuples
save_dict = {
    'name' : img_names,
    'features': embeddings_list
}
torch.save(save_dict, os.path.join(DATASET, f'output/extracted_feature/{'FTed_' if tp == 'a' else ''}{DATA_SOURCE}_embeddings_{BACKBONE}_{backbone_mode}.pt'))
print(f"Final embeddings list shape: {embeddings_list.shape}, saved.")

# %%
if ddp:
    torch.distributed.destroy_process_group()
