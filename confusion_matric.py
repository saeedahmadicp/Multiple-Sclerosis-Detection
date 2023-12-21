import os
import numpy as np
import json
import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
from model import Encoder, SclerosisClassifier
from dataset import reshape_3d, get_train_ds_loader, get_test_ds_loader
from dataset import spliting_data_5_folds
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 12
NUM_EPOCHS = 25
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_DEPTH = 16

DATASET_DIR = 'MS_Dataset'

def main():
    # Reshape the volumes
    reshape = reshape_3d(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, depth=IMAGE_DEPTH)
    def reshape_volume(x): return reshape(x)

    # Transforms
    general_transform = t.Compose([
       t.Lambda(reshape_volume),
    ])
    
    # Split the data into 5 folds
    folds_data = spliting_data_5_folds(DATASET_DIR)
    
    # Data loaders
    for fold_index in range(1):
        train_dl = get_train_ds_loader(dataset_dir=DATASET_DIR, data_dict=folds_data[fold_index], modality="FLAIR", 
                                       transform=general_transform, target_transform=general_transform, batch_size=BATCH_SIZE)
        test_dl = get_test_ds_loader(dataset_dir=DATASET_DIR, data_dict=folds_data[fold_index], modality="FLAIR",
                                     transform=general_transform, target_transform=general_transform, batch_size=BATCH_SIZE)
        
        # Loss function
        loss_fn = nn.MSELoss()
        
        # Define the model
        encoder = Encoder(in_channels=1, filters=[32, 64, 128, 256, 512]).to(DEVICE)
        
        classifier = SclerosisClassifier(input_channels=512, ouptut_units=20).to(DEVICE)  # 20 output units for 20 binary classes
        optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
        
        # Load the encoder weights
        encoder_weights_path = os.path.join('models', 'encoder.pth')
        encoder.load_state_dict(torch.load(encoder_weights_path, map_location=torch.device('cpu')))
        
         ## load the classifier weights
        classifier_weights_path = os.path.join('models', 'classifier.pth')
        classifier.load_state_dict(torch.load(classifier_weights_path, map_location=torch.device('cpu')))
        
        encoder.eval()
        classifier.eval()
        
        # Initialize a list to store confusion matrices for each binary class
        confusion_matrices = [np.zeros((2, 2), dtype=int) for _ in range(20)]  # One confusion matrix for each binary class
        
        with torch.no_grad():
            for batch, (x, _, sp_data) in enumerate(train_dl):  # Usiamo il dataloader di test
                x = x.to(DEVICE).unsqueeze(1)
                
                # Get the features from the encoder
                _, _, _, _, features = encoder(x)
                
                # Forward pass
                sp_features = sp_data['features'].to(DEVICE)
                sp_target = sp_data['target'].to(DEVICE)
                outputs = classifier(features, sp_features)
                
                # Convert the probability to class
                outputs = torch.round(outputs)
                
                # threshold = 0.51
                # outputs[outputs >= threshold] = 1 
                # outputs[outputs < threshold] = 0
                
                
                ## convert output to numpy and integer then
                outputs = outputs.cpu().numpy().astype(int)
                sp_target = sp_target.cpu().numpy().astype(int)
                
                # For each binary class
                for class_i in range(20):
                    # Calculate TP, FP, TN, and FN for each binary class
                    TP = ((outputs[:, class_i] == 1) & (sp_target[:, class_i] == 1)).sum().item()
                    FP = ((outputs[:, class_i] == 1) & (sp_target[:, class_i] == 0)).sum().item()
                    TN = ((outputs[:, class_i] == 0) & (sp_target[:, class_i] == 0)).sum().item()
                    FN = ((outputs[:, class_i] == 0) & (sp_target[:, class_i] == 1)).sum().item()
                    
                    # Update the confusion matrix for the class
                    confusion_matrices[class_i][0, 0] += TP
                    confusion_matrices[class_i][0, 1] += FN
                    confusion_matrices[class_i][1, 0] += FP
                    confusion_matrices[class_i][1, 1] += TN
                    
                    print(f"Class {class_i}: TP={TP}, FP={FP}, TN={TN}, FN={FN}, Confusion Matrix: {confusion_matrices[class_i]}")
                    print(f"Confusion Matrix for Class {class_i}:")
                    plot_confusion_matric(confusion_matrices[class_i], class_i)
                    
                    
                
                
                exit()

        # Now you have a list of correctly calculated confusion matrices, one for each binary class
        # You can print them or visualize them as desired
        # for class_i, confusion_matrix in enumerate(confusion_matrices):
        #     print(f"Confusion Matrix for Class {class_i}:")
        #     print(confusion_matrix)
        #     plot_confusion_matric(confusion_matrix, class_i)
            
def plot_confusion_matric(confusion_matric, class_id):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confusion_matric, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_matric.shape[0]):
        for j in range(confusion_matric.shape[1]):
            ax.text(x=j, y=i, s=confusion_matric[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(f"Confusion Matrix for Class {class_id}", fontsize=18)
    
    ## save the confusion matrix in history folder
    plt.savefig(os.path.join('history', 'confusion_matrix_{}.png'.format(class_id)))
    plt.show()

if __name__ == '__main__':
    main()
