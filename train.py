import os
import numpy as np

import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim


from utils import dice_loss, check_accuracy 
from model import Decoder, Encoder
from dataset import reshape_3d, get_train_ds_loader, get_test_ds_loader
from dataset import visualize_data, spliting_data_5_folds, map_target_values_to_labels


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
NUM_EPOCHS = 100
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_DEPTH = 16

DATASET_DIR = 'dataset'

def main():
    ## reshpae the volumes 
    reshape = reshape_3d(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, depth=IMAGE_DEPTH)
    def reshape_volume(x): return reshape(x)
    ##transforms
    general_transform = t.Compose([
       t.Lambda(reshape_volume),
    ])
    
    ## spliting the data into 5 folds
    folds_data = spliting_data_5_folds(DATASET_DIR)
    
    ## 

