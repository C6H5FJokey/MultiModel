import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
from src.data.data_preprocessing import CustomDataset1
from torch.utils.data import random_split, DataLoader
import os
import matplotlib.pyplot as plt
import gzip
import shutil


def expand_image_to_center(image, target_size=(512, 512)):
    # 获取原图像尺寸
    if not len(image.shape) >= 2:
        return None
    h, w = image.shape[-2:]

    # 计算填充尺寸
    pad_h = (target_size[0] - h) // 2
    pad_w = (target_size[1] - w) // 2

    # 使用 F.pad 进行填充
    padded_image = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

    return padded_image


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class DataAgent:
    def __init__(self, num_blocks=15):
        self.dataset = CustomDataset1(num_blocks)
        self.total_size = total_size = len(self.dataset)
        self.train_size = train_size = int(0.8 * total_size)  # 80% 用于训练
        self.val_size = val_size = int(0.1 * total_size)    # 10% 用于验证
        self.test_size = test_size = total_size - train_size - val_size  # 剩余 10% 用于测试
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
    
    
    def get_train_loader(self, batch_size=8):
        return DataLoader(self.train_dataset, batch_size, shuffle=True)
    
    
    def get_val_loader(self, batch_size=8):
        return DataLoader(self.val_dataset, batch_size, shuffle=False)
    
    
    def get_test_loader(self, batch_size=8):
        return DataLoader(self.test_dataset, batch_size, shuffle=False)


    def plt_idx(self, idx, MRI='T1', show_contour=False, cmap='gray'):
        X, y = self.dataset[idx]
        plt.figure(figsize=(10, 10))
        
        plt.subplot(2, 2, 1)
        plt.title('CT')
        plt.imshow(X[0].reshape((512, 512)), cmap=cmap)
        if show_contour: plt.contour(y, colors='r', linewidths=1)
        
        plt.subplot(2, 2, 2)
        plt.title('PET')
        plt.imshow(X[1].reshape((512, 512)), cmap=cmap)
        if show_contour: plt.contour(y, colors='r', linewidths=1)
        
        plt.subplot(2, 2, 3)
        if MRI == 'T1': 
            plt.title('T1')
            plt.imshow(X[2].reshape((512, 512)), cmap=cmap)
            if show_contour: plt.contour(y, colors='r', linewidths=1)
        else:
            plt.title('T2')
            plt.imshow(X[3].reshape((512, 512)), cmap=cmap)
            if show_contour: plt.contour(y, colors='r', linewidths=1)
        
        plt.subplot(2, 2, 4)
        plt.imshow(y.reshape((512, 512)), cmap='gray')
    
    

def save_model_incrementally(net, directory, base_name='net'):
    # 确保目录存在
    os.makedirs(directory, exist_ok=True)
    
    # 获取现有文件的编号
    existing_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith('.pth')]
    
    # 提取已有文件的编号
    numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if '_' in f]
    
    # 确定下一个编号
    next_number = max(numbers) + 1 if numbers else 1
    
    # 构建文件名
    file_name = f"{base_name}_{next_number}.pth"
    file_path = os.path.join(directory, file_name)
    
    # 保存模型
    torch.save(net.state_dict(), file_path)
    print(f"Model saved as {file_name}")
    

def log_dice_ce_loss_with_logit(y_hat, y, ep=1e-8):
    if len(y.shape) == 3:
        y = y.unsqueeze(1)
    ce_loss = nn.BCEWithLogitsLoss(reduction='none')
    pixel_wise_ce = ce_loss(y_hat, y)
    y_hat = torch.sigmoid(y_hat)
    
    union = torch.clamp(y_hat + y, min=0, max=1)
    union_area = torch.sum(union, dim=(1, 2, 3))
    
    weighted_ce = torch.sum(pixel_wise_ce * union, dim=(1, 2, 3)) / (union_area + ep)
    ce_total_loss = weighted_ce.mean()
    
    intersection = torch.sum(y_hat * y, dim=(1, 2, 3))
    y_hat_sum = torch.sum(y_hat, dim=(1, 2, 3))
    y_sum = torch.sum(y, dim=(1, 2, 3))
    dice = (2. * intersection + ep) / (y_hat_sum + y_sum + ep)
    dice = torch.clamp(dice, min=ep, max=1.0)
    dice_loss = -torch.log(dice).mean()
    dice_ce_loss = dice_loss + ce_total_loss
    return dice_ce_loss


def log_dice_loss_with_logit(y_hat, y, ep=1e-8):
    if len(y.shape) == 3:
        y = y.unsqueeze(1)
    ce_loss = nn.BCEWithLogitsLoss(reduction='none')
    pixel_wise_ce = ce_loss(y_hat, y)
    # ce_total_loss = pixel_wise_ce.mean()
    weighted_ce = pixel_wise_ce.mean((1, 2, 3))
    
    # if y.sum() == 0: return ce_total_loss * 2
    mask = y.sum((1,2,3)) != 0
    
    y_hat = torch.sigmoid(y_hat)
    intersection = torch.sum(y_hat * y, dim=(1, 2, 3))
    y_hat_sum = torch.sum(y_hat, dim=(1, 2, 3))
    y_sum = torch.sum(y, dim=(1, 2, 3))
    dice = (2. * intersection + ep) / (y_hat_sum + y_sum + ep)
    dice = torch.clamp(dice, min=ep, max=1.0)
    dice_loss = -torch.log(dice)

    total_loss = torch.zeros_like(dice_loss)
    total_loss[mask] = dice_loss[mask] + weighted_ce[mask]
    total_loss[~mask] = weighted_ce[~mask]
    
    total_loss = total_loss.mean()
    
    return total_loss


def custom_loss(y_hat, y, alpha=0.25, gamma=2, ep=1e-8):
    """
    自定义损失函数，结合Dice损失和Focal Loss，处理前景和背景不平衡问题。
    
    参数:
    y_hat (Tensor): 模型的预测值， logits 形式。
    y (Tensor): 真实标签。
    alpha (float): Focal Loss中的平衡因子。
    gamma (float): Focal Loss中的聚焦参数。
    ep (float): 防止数值不稳定的小数。
    
    返回:
    Tensor: 批量损失的均值。
    """
    # 检查标签维度，确保是4维
    if len(y.shape) == 3:
        y = y.unsqueeze(1)
    
    # 计算Focal Loss
    p = torch.sigmoid(y_hat)
    focal_loss = -alpha * y * (1 - p)**gamma * torch.log(p + ep) \
                 - (1 - alpha) * (1 - y) * p**gamma * torch.log(1 - p + ep)
    
    # 计算Dice损失
    intersection = torch.sum(p * y, dim=(1, 2, 3))
    y_sum = torch.sum(y, dim=(1, 2, 3))
    p_sum = torch.sum(p, dim=(1, 2, 3))
    dice = (2. * intersection + ep) / (p_sum + y_sum + ep)
    dice_loss = -torch.log(dice)
    
    # 判断样本是否有前景
    has_foreground = y.sum(dim=(1, 2, 3)) > 0
    
    # 组合损失
    total_loss = torch.zeros_like(dice_loss)
    total_loss[has_foreground] = dice_loss[has_foreground] + focal_loss[has_foreground].mean(dim=(1, 2, 3))
    total_loss[~has_foreground] = focal_loss[~has_foreground].mean(dim=(1, 2, 3))
    
    # 批量平均损失
    total_loss = total_loss.mean()
    return total_loss


def ce_loss_with_logit(y_hat, y, ep=1e-8):
    if len(y.shape) == 3:
        y = y.unsqueeze(1)
    ce_loss = nn.BCEWithLogitsLoss(reduction='none')
    pixel_wise_ce = ce_loss(y_hat, y)
    
    y_hat = torch.sigmoid(y_hat)
    union = torch.clamp(y_hat + y, min=0, max=1)
    union_area = torch.sum(union, dim=(1, 2, 3))
    
    weighted_ce = torch.sum(pixel_wise_ce * union, dim=(1, 2, 3)) / (union_area + ep)
    ce_total_loss = weighted_ce.mean()
    return ce_total_loss
    


def bin_dice_eval_with_logit(y_hat, y, threshold=0.5, ep=1e-8):
    if len(y.shape) == 3:
        y = y.unsqueeze(1)
    y_hat = (torch.sigmoid(y_hat) > threshold).float()
    intersection = torch.sum(y_hat * y, dim=(1, 2, 3))
    y_hat_sum = torch.sum(y_hat, dim=(1, 2, 3))
    y_sum = torch.sum(y, dim=(1, 2, 3))
    union = y_hat_sum + y_sum
    return (intersection * 2. + ep) / (union + ep)


def save_compressed_checkpoint(state, directory, filename):
    os.makedirs(directory, exist_ok=True)
    torch.save(state, os.path.join(directory, filename) + '.temp')
    with open(os.path.join(directory, filename) + '.temp', 'rb') as f_in:
        with gzip.open(os.path.join(directory, filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(os.path.join(directory, filename) + '.temp')

def load_compressed_checkpoint(directory, filename):
    with gzip.open(os.path.join(directory, filename), 'rb') as f:
        return torch.load(f)