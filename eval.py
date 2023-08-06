import json
import os

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = ['dice_coeff', 'multiclass_dice_coeff', 'calculate_dice_score']


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
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

    target = F.one_hot(target, 4).permute(0,4,1,2,3).float()
    input = F.one_hot(preds.argmax(1), 4).permute(0,4,1,2,3).float()

    input = input[:, 1:, :, :, :]
    target = target[:, 1:, :, :, :]

    assert input.size() == target.size()
    dice = 0
    all_dice_score = []
    for channel in range(input.shape[1]):
        classwise_dice = dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        all_dice_score.append(classwise_dice)
        dice += classwise_dice
    
    return  dice
    




def calculate_dice_score(encoder, decoder, loader, device, save_results=False, epoch=0):
    
    dice_dict = {}
    dice_dict['mean'] = 0.0
    
    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for  x, y in loader:
            x = x.to(device)
            y = y.to(device)
  
            x1, x2, x3, x4, x5 = encoder(x)
            output = decoder(x1, x2, x3, x4, x5)
            
            preds = torch.softmax(output)
            batch_dict = multiclass_dice_coeff(preds=preds, target=y)
            dice_dict['mean'] += batch_dict['mean']
            
    

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