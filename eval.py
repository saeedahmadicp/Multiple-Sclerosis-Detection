import json
import os

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = ['dice_coeff', 'multiclass_dice_coeff', 'calculate_dice_score']


def dice_coeff(prediction, target, epsilon=1e-6):
    """
    Calculate the Dice coefficient matrix for binary segmentation pixel volumes.

    Args:
        prediction (torch.Tensor): Binary segmentation prediction volume with shape [B, C, H, W, D].
        target (torch.Tensor): Binary segmentation target volume with shape [B, C, H, W, D].
        epsilon (float, optional): Small constant to avoid division by zero. Default is 1e-6.

    Returns:
        torch.Tensor: Dice coefficient matrix with shape [B, C].
    """
    intersection = torch.sum(prediction * target, dim=(2, 3, 4))
    prediction_sum = torch.sum(prediction, dim=(2, 3, 4))
    target_sum = torch.sum(target, dim=(2, 3, 4))
    
    dice_coefficient = (2.0 * intersection + epsilon) / (prediction_sum + target_sum + epsilon)
    
    return dice_coefficient

def multiclass_dice_coeff(preds: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes

    p = 1
    smooth = 1
    probs = torch.sigmoid(preds)
    numer = (probs * target).sum() # Union (A or B)
    denor = (probs.pow(p) + target.pow(p)).sum() # A + B
    dice_score = (2 * numer + smooth) / (denor + smooth)
    
    
    return  dice_score
    


def calculate_dice_score(encoder, decoder, loader, device, save_results=False, epoch=0):
    
    dice_dict = {}
    dice_dict['mean'] = 0.0
    
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for  x, y, _ in loader:
            x = x.to(device).unsqueeze(1)
            y = y.to(device)
  
            x1, x2, x3, x4, x5 = encoder(x)
            output = decoder(x1, x2, x3, x4, x5)
            
            batch_dict = multiclass_dice_coeff(preds=output, target=y)
            dice_dict['mean'] += batch_dict
            
    

    dice_dict['mean'] /= len(loader)
    

    dice_dict['mean'] = dice_dict['mean'].detach().cpu().item()


    if save_results:
        if not os.path.exists(f"./results/"):
            os.makedirs(f"./results/")
        json_file = json.dumps(dice_dict)
        f = open(f"./results/epoch_{epoch}.json", "w")
        f.write(json_file)
        f.close()

    return dice_dict