import os
from torch.utils.data import Dataset
import nibabel as nib
import matplotlib.pyplot as plt


class MultipleSclerosisDataset(Dataset): # Dataset class
    def __init__(self, dataset_dir, data_dict, data_type, modality="FLAIR",transform=None, target_transform=None):
        self.dataset_dir = dataset_dir
        self.modality = modality
        self.transform = transform
        self.target_transform = target_transform
        
        # samples list
        self.samples = data_dict[f'{data_type}_samples']
        
    def __len__(self):
        return len(self.samples)
    
    
    def __getitem__(self, index):
        # load the sample
        sample = sample.split('-')[1]
        # load the image and the mask
        volume = nib.load(os.path.join(self.dataset_dir, f'Patient-{sample}' , f'{sample}-{self.modality}.nii')).get_fdata()
        mask = nib.load(os.path.join(self.dataset_dir, f'Patient-{sample}' , f'{sample}-LesionSeg-{self.modality}.nii')).get_fdata()
        
        # apply transforms
        if self.transform:
            volume = self.transform(volume)
        if self.target_transform:
            mask = self.target_transform(mask)
            
        return volume, mask
        
        
## function for visualizing the data (23 slices) in grid view
def visualize_data(data, slices, figure_size=(10, 10)):
    plt.figure(figsize=figure_size)
    for i in range(slices):
        plt.subplot(5, 5, i+1)
        plt.imshow(data[:, :, i], cmap='gray')
        plt.axis('off')
    plt.show()
    
    
if __name__ == '__main__':
    path = 'Dataset'
    modalities = ['FLAIR', 'T1', 'T2']
    index = 3
    data = nib.load(os.path.join(path, f'Patient-{index}' , f'{index}-{modalities[0]}.nii')).get_fdata()
    slices = data.shape[2]
    print(data.shape)
    visualize_data(data, slices)




