import torch
import numpy as np


def find_class_wise_accuracies(data_dl, classifier, encoder, device):
    ## calculate the accuracy for 20 classes
    
    overall_accuracies = np.zeros(20)
    
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for batch, (x, _, sp_data) in enumerate(data_dl):
            x = x.to(device).unsqueeze(1)
        
            ## get the features from the encoder
            _, _, _, _, features = encoder(x)
        
            ## forward pass
            sp_features = sp_data['features'].to(device)
            sp_target = sp_data['target'].to(device)
            outputs = classifier(features, sp_features)
        
            ## convert the probability to class
            outputs = torch.round(outputs)
        
            ## batch-wise accuracy
            accuracy = []
            ## check the accuracy for each label sperately, muliti-class multi-label
            for class_i in range(20):
                accuracy.append((outputs[:, class_i] == sp_target[:, class_i]).sum().item())
                overall_accuracies[class_i] += accuracy[class_i]
        
    return overall_accuracies / len(data_dl)