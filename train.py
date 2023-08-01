import os
import numpy as np
import json

import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim


from utils import  check_accuracy, Fit, DiceBCELossLogitsLoss
from model import Decoder, Encoder
from dataset import reshape_3d, get_train_ds_loader, get_test_ds_loader
from dataset import visualize_data, spliting_data_5_folds, map_target_values_to_labels


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_EPOCHS = 100
IMAGE_HEIGHT =  256
IMAGE_WIDTH =   256
IMAGE_DEPTH = 16

DATASET_DIR = 'MS_Dataset'

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
    
    ## data loaders
    for fold_index in range(1):
        train_dl = get_train_ds_loader(dataset_dir=DATASET_DIR, data_dict=folds_data[fold_index], modality="FLAIR", 
                                       transform=general_transform, target_transform=general_transform, batch_size=BATCH_SIZE)
        test_dl = get_test_ds_loader(dataset_dir=DATASET_DIR, data_dict=folds_data[fold_index], modality="FLAIR",
                                     transform=general_transform, target_transform=general_transform, batch_size=BATCH_SIZE)
        
        
        ## loss function
        loss_fn = DiceBCELossLogitsLoss()
        
        ## define the model
        encoder = Encoder(in_channels=1, filters=[32, 64, 128, 256, 512]).to(DEVICE)
        decoder = Decoder(out_channels=1, filters=[32, 64, 128, 256, 512]).to(DEVICE)
        
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
        history = Fit(train_dl, test_dl, encoder, decoder, optimizer, loss_fn, DEVICE, epochs=NUM_EPOCHS)
        
        ## save the history in json file
        os.makedirs('history', exist_ok=True)
        history_path = os.path.join('history', f'history_fold_{fold_index}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
            f.close()
        
        print(f"Fold {fold_index} is done!")
        print()
        

if __name__ == '__main__':
    main()