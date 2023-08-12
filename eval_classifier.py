import os
import numpy as np
import json

import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim


from model import  Encoder, SclerosisClassifier
from dataset import reshape_3d, get_train_ds_loader, get_test_ds_loader
from dataset import  spliting_data_5_folds, map_target_values_to_labels
from classifier_utils import find_class_wise_accuracies

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
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
        test_dl = get_test_ds_loader(dataset_dir=DATASET_DIR, data_dict=folds_data[fold_index], modality="FLAIR",
                                     transform=general_transform, target_transform=general_transform, batch_size=BATCH_SIZE)
        
        
        ## define the models
        encoder = Encoder(in_channels=1, filters=[32, 64, 128, 256, 512]).to(DEVICE) ## encoder
        classifier = SclerosisClassifier(input_channels=512, ouptut_units=20).to(DEVICE) ## classifier
        
        
        ## load the encoder weights
        encoder_weights_path = os.path.join('models', 'encoder.pth')
        encoder.load_state_dict(torch.load(encoder_weights_path))
        
        ## load the classifier weights
        classifier_weights_path = os.path.join('models', 'classifier.pth')
        classifier.load_state_dict(torch.load(classifier_weights_path))
        
       

        
        ## test the model
        accuracies = find_class_wise_accuracies(classifier=classifier, encoder=encoder,data_dl=test_dl, device=DEVICE)
        
        ## map the target values to labels
        values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19]
        dict_map = map_target_values_to_labels(values=values, dataset_dir=DATASET_DIR)
        
        ## print the accuracies
        for index, (key, value) in enumerate(dict_map.items()):
            print("Class: {}, Label: {}, Accuracy: {}".format(key, value, (round(accuracies[index] *100, 2))))
        
        print("\nOverall Mean Accuracy: {}\n\n".format(round(np.mean(accuracies), 2)*100))

   
            

if __name__ == '__main__':
    main()