import torch
import torch.nn.functional as F
import random
import numpy as np
from src.data.data_preprocessing import CustomDataset1
from torch.utils.data import random_split, DataLoader
import os
import matplotlib.pyplot as plt


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
    def __init__(self, num_blocks=12):
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
        plt.imshow(X[0], cmap='gray')
        if show_contour: plt.contour(y, colors='r', linewidths=1)
        
        plt.subplot(2, 2, 2)
        plt.title('PET')
        plt.imshow(X[1], cmap='gray')
        if show_contour: plt.contour(y, colors='r', linewidths=1)
        
        plt.subplot(2, 2, 3)
        if MRI == 'T1': 
            plt.title('T1')
            plt.imshow(X[2], cmap='gray')
            if show_contour: plt.contour(y, colors='r', linewidths=1)
        else:
            plt.title('T2')
            plt.imshow(X[3], cmap='gray')
            if show_contour: plt.contour(y, colors='r', linewidths=1)
        
        plt.subplot(2, 2, 4)
        plt.imshow(y, cmap='gray')
    

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