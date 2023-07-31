import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


__all__ = ['DiceBCELossLogitsLoss', 'evaluate', 'dice_coeff', 'check_accuracy', 'train_one_epoch', 'Fit']

class DiceBCELossLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELossLogitsLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def dice_coeff(pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def evaluate(preds, targets):
    """ 
      Returns specificty, precision, recall and f1_score   

    """

    confusion_vector = preds / targets
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    ### precision, recall, f1_score and specificity
    specificity = true_negatives / ((true_negatives + false_positives) + 1e-7)
    precision = true_positives / ((true_positives + false_positives) + 1e-7)
    recall = true_positives / ((true_positives + false_negatives) + 1e-7)
    f1_score = (2.0 * (recall*precision)) / ((recall + precision)+ 1e-7)

    dict = {
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return dict

def dice_coeff(pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def check_accuracy(loader, encoder, decoder, device, threshold=0.5, test=False):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    
    encoder.eval()
    decoder.eval()
    if test:
        f1_score, precision, recall, specificity = 0.0, 0.0 , 0.0 , 0.0

    with torch.no_grad():
        for _, (x, y, _) in enumerate(loader):
            
            x = x.to(device).unsqueeze(1)
            y = y.to(device) #.unsqueeze(1)

            ## forward pass
            x1, x2, x3, x4, x5 = encoder(x)
            preds = decoder(x1, x2, x3, x4, x5)

            preds = (preds > threshold).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += dice_coeff(preds, y)

            if test:
                temp_dict = evaluate(preds, y)
                f1_score += temp_dict['f1_score']
                precision += temp_dict['precision']
                recall += temp_dict['recall']
                specificity += temp_dict['specificity']



        

        accuracy = num_correct/num_pixels*100
        dice_score = (dice_score/(len(loader)))*100
        
        accuracy, dice_score = accuracy.detach().cpu().item() , dice_score.detach().cpu().item() 

    if test: 
        f1_score = (f1_score/(len(loader)))*100
        precision = (precision/(len(loader)))*100
        recall = (recall/(len(loader)))*100
        specificity = (specificity/(len(loader)))*100

        dict = {
                'specificity': specificity,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy':accuracy,
                'dice_score': dice_score,
                }
        return dict

    else:
        print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}" )
        print(f"Dice score: {dice_score :.2f}")
    
        return accuracy, dice_score
    
    
def train_one_epoch(train_dl, encoder, decoder, optimizer, loss_fn, device):
    mean_loss = 0.0
        
    for i, (x, y, _) in enumerate(train_dl):
        x = x.to(device).unsqueeze(1)
        y = y.to(device)
            
        x1, x2, x3, x4, x5 = encoder(x)
        out = decoder(x1, x2, x3, x4, x5)
            
        loss = loss_fn(y, out)
        mean_loss += loss.detach().cpu().item()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if i % 10 == 0:
            print(f"Iteration {i+1} of {len(train_dl)}")
            print(f"Train Loss: {loss:.4f}")
            print()
            
    return mean_loss / len(train_dl)

def Fit(train_dl, test_dl, encoder, decoder, optimizer, loss_fn, device, epochs=100):
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_dice_scores = []
    
    
    best_f1_score = 0.0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_loss = train_one_epoch(train_dl, encoder, decoder, optimizer, loss_fn, device)
        test_dict = check_accuracy(test_dl, encoder, decoder, device, test=True)
        
        
        train_losses.append(train_loss)
        
        test_f1_score = test_dict['f1_score']
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test F1 Score: {test_dict['f1_score']:.4f}")
        print(f"Test Precision: {test_dict['precision']:.4f}")
        print(f"Test Recall: {test_dict['recall']:.4f}")
        print(f"Test Specificity: {test_dict['specificity']:.4f}")
        print()
        
        ## check if the current model is the best model, if so save it
        os.makedirs('models', exist_ok=True)
        if test_f1_score > best_f1_score:
            best_f1_score = test_f1_score
            encoder_path = os.path.join('models', 'encoder.pth')
            decoder_path = os.path.join('models', 'decoder.pth')
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
        
    
    history = { 'train_loss': train_losses,
                'test_loss': test_losses,
                'test_accuracy': test_accuracies,
                'test_dice_score': test_dice_scores }
     
    return history
