import torch
import torch.nn as nn

__all__ = ['CombinedLoss']

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0, epsilon=1e-6):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for Dice Loss
        self.beta = beta    # Weight for BCE Loss
        self.gamma = gamma  # Weight for Focal Loss
        self.epsilon = epsilon
        self.dice_loss = DiceLoss(epsilon)
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, prediction, target):
        dice_loss = self.dice_loss(prediction, target)
        bce_loss = self.bce_loss(torch.sigmoid(prediction), target.float())
        focal_loss = self.focal_loss(prediction, target.float())
        
        combined_loss = self.alpha * dice_loss + self.beta * bce_loss + self.gamma * focal_loss
        return combined_loss * 1/3

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target)
        dice_score = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        loss = 1.0 - dice_score
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, prediction, target):
        sigmoid_pred = torch.sigmoid(prediction)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()


