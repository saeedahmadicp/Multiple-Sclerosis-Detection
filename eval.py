import json
import os

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = ['dice_coeff', 'multiclass_dice_coeff', 'calculate_dice_score']


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    input = input.squeeze(1)  # B,1,H,W => B,H,W
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


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
    

    print(f"dice mean score: {dice_dict['mean']}")
  

    if save_results:
        if not os.path.exists(f"./results/"):
            os.makedirs(f"./results/")
        json_file = json.dumps(dice_dict)
        f = open(f"./results/epoch_{epoch}.json", "w")
        f.write(json_file)
        f.close()

    return dice_dict