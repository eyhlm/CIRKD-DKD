import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
program_started_logged = False

__all__ = ['DKD']

def replace_negative_one_gt(logits,tensor):
    # 使用 torch.where 将 -1 替换为 1
    result_tensor = torch.where(tensor == -1, torch.tensor(logits.size(1)).to(tensor.device), tensor)
    return result_tensor

def replace_negative_one_other(logits,tensor):
    # 使用 torch.where 将 -1 替换为 0-10
    result_tensor = torch.where(tensor == -1, torch.arange(0, logits.size(1)).to(tensor.device), tensor)
    return result_tensor

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    # print(target.unsqueeze(1))
    new_column = torch.zeros((logits.size(0), 1)).to(torch.device("cuda"))
    mask = torch.cat([torch.zeros_like(logits), new_column], dim=1)
    mask = torch.cat([torch.zeros_like(logits), new_column], dim=1)
    label_copy = replace_negative_one_gt(logits,target.unsqueeze(1))
    # print(label_copy)
    # print(mask)
    mask = mask.scatter_(1, label_copy, 1).bool()
    # print(mask)

    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    # print(target.unsqueeze(1))
    mask = torch.ones_like(logits)
    label_copy = replace_negative_one_other(logits,target.unsqueeze(1))
    # print(label_copy)
    # print(mask)
    mask = mask.scatter_(1, label_copy, 0).bool()
    # print(mask)

    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1[:,0:t.size(1)]).sum(dim=1, keepdims=True)
    t2 = (t * mask2[:,0:t.size(1)]).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    # print(rt)
    return rt

def conut_useful_num_target(target):
    target = target.reshape(-1)
    target = target.unsqueeze(1)
    count_negative_ones = torch.sum(torch.eq(target, -1))
    result = target.size(0) - count_negative_ones
    return result


class DKD(nn.Module):
    '''
    解耦知识蒸馏损失
    '''
    def __init__(self, temperature=1, alpha=1,beta=8):
        super(DKD, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred, soft,targets):
        global program_started_logged
        #先将教师输出和学生输出上采样到原图片大小以适配target大小
        B_targets, H_targets, W_targets = targets.size()
        useful_num_target =conut_useful_num_target(targets)
        if not program_started_logged:
            logging.info(f"B_targets:{B_targets} H_targets:{H_targets} W_targets:{W_targets}")
        
        pred_interpolate = F.interpolate(pred, (H_targets, W_targets), mode='bilinear', align_corners=True)
        soft_interpolate = F.interpolate(soft, (H_targets, W_targets), mode='bilinear', align_corners=True)

        B, C, h, w = soft.size()
        logits_student = pred_interpolate.permute(0,2,3,1).contiguous().view(-1,C)
        logits_teacher = soft_interpolate.permute(0,2,3,1).contiguous().view(-1,C)
        
        gt_mask = _get_gt_mask(logits_student, targets)
        other_mask = _get_other_mask(logits_teacher, targets)

        # 先cat再softmax确保标签为-1时，两个分布时相同且不为零的
        logits_teacher_tckd = cat_mask(logits_teacher, gt_mask, other_mask)
        logits_student_tckd = cat_mask(logits_student, gt_mask, other_mask)
        softmax_student_tckd = F.softmax(logits_student_tckd / self.temperature, dim=1)
        softmax_teacher_tckd = F.softmax(logits_teacher_tckd / self.temperature, dim=1)

        tckd_loss = (
            F.kl_div(torch.log(softmax_student_tckd), softmax_teacher_tckd, reduction='sum')
            * (self.temperature**2)
            / useful_num_target
        )

        softmax_teacher_nckd = F.softmax(
            logits_teacher * (other_mask).int() / self.temperature - 100.0 * gt_mask[:,0:logits_teacher.size(1)], dim=1
        )
        log_softmax_student_nckd = F.log_softmax(
            logits_student * (other_mask).int() / self.temperature - 100.0 * gt_mask[:,0:logits_student.size(1)], dim=1
        )

        nckd_loss = (
            F.kl_div(log_softmax_student_nckd, softmax_teacher_nckd, reduction='sum')
            * (self.temperature**2)
            / useful_num_target
        )
        
        # p_s = F.log_softmax(scale_pred / self.temperature, dim=1)
        # p_t = F.softmax(scale_soft / self.temperature, dim=1)
        # loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.temperature**2)
        if not program_started_logged:
            logging.info(f"useful_num_target:{useful_num_target}\n")
            logging.info(f"pred_interpolate size:\n{pred_interpolate.shape}")
        program_started_logged = True
        logging.info(f"tckd_loss:{tckd_loss} nckd_loss:{nckd_loss}\n")
        return self.alpha * tckd_loss + self.beta * nckd_loss