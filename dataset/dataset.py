import os

import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate


__all__ = ['MultipleSclerosisDataset', 'reshape_3d', 'visualize_data', 'spliting_data_5_folds', 'map_target_values_to_labels',]


class MultipleSclerosisDataset(Dataset): # Dataset class
    def __init__(self, dataset_dir, data_dict, data_type, modality="FLAIR",transform=None, target_transform=None):
        self.dataset_dir = dataset_dir
        self.modality = modality
        self.transform = transform
        self.target_transform = target_transform
        
        ## suplementary data
        self.supplementory_data = preprocess_supplementory_data(dataset_dir)
        
        
        # samples list
        self.samples = data_dict[f'{data_type}_samples']
        
    def __len__(self):
        return len(self.samples)
    
    
    def __getitem__(self, index):
        # load the sample
        sample = self.samples[index].split('-')[1]
        # load the image and the mask
        volume = nib.load(os.path.join(self.dataset_dir, f'Patient-{sample}' , f'{sample}-{self.modality}.nii')).get_fdata()
        mask = nib.load(os.path.join(self.dataset_dir, f'Patient-{sample}' , f'{sample}-LesionSeg-{self.modality}.nii')).get_fdata()
        
        ## converting the data to tensor
        volume = torch.from_numpy(volume).float()
        mask = torch.from_numpy(mask).float()
        
        ## normalizing the data to max-min 
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        ## apply transforms
        if self.transform:
            volume = self.transform(volume)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        ## change the data type of mask to long
        mask = mask.long()
        
        ## supplementory data dictionary for the current sample
        supplementory_data_dict = self.supplementory_data[int(sample)-1]
        
        return volume, mask, supplementory_data_dict
        
## function for reshaping the 3D data
class reshape_3d(torch.nn.Module):   
    def __init__(self, height, width, depth, mode='nearest'):
        super(reshape_3d, self).__init__()
        self.height = height
        self.width = width
        self.depth = depth
        self.mode = mode

    def forward(self, x):
        
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
            x = interpolate(x,size=(self.height, self.width, self.depth), mode=self.mode, )
            x = x.squeeze(0)
        else:
            x = x.unsqueeze(0).unsqueeze(0)
            x = interpolate(x,size=(self.height, self.width, self.depth), mode=self.mode, )
            x = x.squeeze(0).squeeze(0)
        return x
    
 
## function for visualizing the data (23 slices) in grid view
def visualize_data(data, slices, figure_size=(10, 10)):
    plt.figure(figsize=figure_size)
    for i in range(slices):
        plt.subplot(4, 4, i+1)
        plt.imshow(data[:, :, i], cmap='gray')
        plt.axis('off')
    plt.show()

def spliting_data_5_folds(dataset_dir):
   
    folds_data = []
    folders = os.listdir(dataset_dir)
    
    ## filter the folders to get only the patients folders
    folders = list(filter(lambda x: x.startswith('Patient'), folders))

    kfold = KFold(n_splits=5, shuffle=True, random_state=20)

    indices = kfold.split(folders, folders)

    for train_indices, valid_indices in indices:
        train_samples = [folders[index] for index in train_indices]
        valid_samples = [folders[index] for index in valid_indices]

        folds_data.append({
            "train_samples": train_samples,
            "valid_samples": valid_samples,
        })

    return folds_data

def map_target_values_to_labels(values, dataset_dir):
    labels = []
    patients_info = pd.read_excel(os.path.join(dataset_dir, 'Supplementary Table 1 for patient info .xlsx'), header=1)
    
    columns_PI = patients_info.columns
    labels = patients_info[columns_PI[-20:]]
    
    map_dict = {}
    for i in range(len(labels.columns)):
        map_dict[labels.columns[i]] = values[i]
    return map_dict

def preprocess_supplementory_data(dataset_dir):
    patients_info = pd.read_excel(os.path.join(dataset_dir, 'Supplementary Table 1 for patient info .xlsx'), header=1)
    
    columns_PI = patients_info.columns
    
    ## target labels, last 20 columns, drop these columns from the dataframe
    target = patients_info[columns_PI[-20:]]
    patients_info.drop(columns=columns_PI[-20:], inplace=True)
    
    ## label encoding for gender column 
    patients_info[columns_PI[1]] = patients_info[columns_PI[1]].map({'F': 0, 'M': 1})
    ## label encoding for 6th column, Yes: 1, No: 0, question: Does the time difference between MIR aquisition and EDSS < two months?
    patients_info[columns_PI[5]] = patients_info[columns_PI[5]].map({'Yes': 1, 'No': 0})
    ## label encoding for 9th column, Yes: 1, No: 0, question: Does the patient has co-moroidity?
    patients_info[columns_PI[8]] = patients_info[columns_PI[8]].map({'Yes': 1, 'No': 0})
    
    ## one-hot encoding for 7th and 8th columns
    patients_info = pd.get_dummies(patients_info, columns=[columns_PI[6], columns_PI[7]])
    
    supplementory_data =  []
    
    for sample in range(len(patients_info)):
        sample_dict = {}
        # for each sample, get three keys, patient_id, target, and features
        sample_dict['patient_id'] = patients_info[columns_PI[0]][sample]
        sample_dict['target'] = target.iloc[sample].values
        ## all features except the patient_id
        sample_dict['features'] = patients_info.iloc[sample].values[1:]
        supplementory_data.append(sample_dict)
        
    return supplementory_data
    
    

# if __name__ == '__main__':
#     path = 'MS_Dataset'
    
#     reshape = reshape_3d(128,128,16)
#     def reshape_volume(x): return reshape(x)
#     general_transforms = t.Compose([reshape_volume])
    
#     data_dict = spliting_data_5_folds(path)
#     sp_data = preprocess_supplementory_data(path)

    
#     ds = MultipleSclerosisDataset(path, data_dict[0], 'train', transform=general_transforms, target_transform=general_transforms)
    
#     x, y, sp_data = ds[0]
#     print(x.shape, y.shape)
#     print(sp_data)
   
    
   
    




