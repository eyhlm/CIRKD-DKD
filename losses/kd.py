import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
program_started_logged = False

__all__ = ['CriterionKD']

class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''
    def __init__(self, temperature=1):
        super(CriterionKD, self).__init__()
        self.temperature = temperature

    def forward(self, pred, soft):
        global program_started_logged
        B, C, h, w = soft.size()
        scale_pred = pred.permute(0,2,3,1).contiguous().view(-1,C)
        scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C)
        p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
        p_t = F.softmax(scale_soft / self.temperature, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        if not program_started_logged:
            logging.info(f"soft size:\n{soft.shape}")
            logging.info(f"pred size:\n{pred.shape}")
            logging.info(f"soft.permute(0,2,3,1) size:\n{soft.permute(0,2,3,1).shape}")
            logging.info(f"scale_soft = soft.permute(0,2,3,1).contiguous().view(-1,C) size:\n{soft.permute(0,2,3,1).contiguous().view(-1,C).shape}")
        program_started_logged = True
        return loss