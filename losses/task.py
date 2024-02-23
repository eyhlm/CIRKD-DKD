import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

__all__ = ['SegCrossEntropyLoss']
program_started_logged = False
class SegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, **kwargs):
        super(SegCrossEntropyLoss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        global program_started_logged
        B, H, W = targets.size()
        inputs = F.interpolate(inputs, (H, W), mode='bilinear', align_corners=True)
        if not program_started_logged:
            logging.info(f"input after interpolate:\n{inputs.shape}")
        program_started_logged = True
        return self.task_loss(inputs, targets)