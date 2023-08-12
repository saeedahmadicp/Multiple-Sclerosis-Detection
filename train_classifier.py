import os
import numpy as np
import json

import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim


from model import  Encoder, SclerosisClassifier
from dataset import reshape_3d, get_train_ds_loader, get_test_ds_loader
from dataset import  spliting_data_5_folds
from classifier_utils import find_class_wise_accuracies

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
NUM_EPOCHS = 50
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
        loss_fn = nn.MSELoss()
        
        ## define the model
        encoder = Encoder(in_channels=1, filters=[32, 64, 128, 256, 512]).to(DEVICE)
        
        classifier = SclerosisClassifier(input_channels=512, ouptut_units=20).to(DEVICE)
        optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
        
        ## load the encoder weights
        encoder_weights_path = os.path.join('models', 'encoder.pth')
        encoder.load_state_dict(torch.load(encoder_weights_path))
        
        for param in encoder.parameters():
            param.requires_grad = False
        
        
        
        for epoch in range(NUM_EPOCHS):
            
            for batch, (x, _, sp_data) in enumerate(train_dl):
                x = x.to(DEVICE).unsqueeze(1)
            
            
                ## get the features from the encoder
                _, _, _, _, features = encoder(x)
            
                ## forward pass
                sp_features = sp_data['features'].to(DEVICE)
                sp_target = sp_data['target'].to(DEVICE)
                outputs = classifier(features, sp_features)
            
                ## calculate the loss
                loss = loss_fn(outputs, sp_target)
            
                ## backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                
            
                ##culculate the accuracy for 20 classes
                accuracy = []
            
                ## convert the probability to class
                outputs = torch.round(outputs)
                
                if batch % 10 == 0:
                    ## check the accuracy for each label sperately, muliti-class multi-label
                    for class_i in range(20):
                        accuracy.append((outputs[:, class_i] == sp_target[:, class_i]).sum().item())
                    
                    print('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, Accuracy: {}'.format(epoch+1, NUM_EPOCHS, batch+1, len(train_dl), loss.item(), np.mean(accuracy)))
                    #print(np.mean(accuracy))
            
            ## save the model
            path = os.path.join('models', 'classifier.pth')
            torch.save(classifier.state_dict(), path)
        
            





if __name__ == '__main__':
    main()