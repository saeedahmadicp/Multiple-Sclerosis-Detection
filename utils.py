import numpy as np

import torch
import torch.nn.functional as F

__all__ = ['dice_loss', 'evaluate', 'dice_coeff', 'check_accuracy']

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W, D].
        logits: a tensor of shape [B, C, H, W, D]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:

        true_1_hot = torch.eye(num_classes)[true.squeeze(1).cpu()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()

        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

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
    specificity = true_negatives / (true_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = (2.0 * (recall*precision)) / (recall + precision)

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

def check_accuracy(loader, model, device="cuda", threshold=0.5, test=False):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    if test:
        f1_score, precision, recall, specificity = 0.0, 0.0 , 0.0 , 0.0

    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            
            x = x.to(device)
            y = y.to(device) #.unsqueeze(1)

            ## for unet plus plus  
            preds = torch.sigmoid((model(x)))

            # for unet

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