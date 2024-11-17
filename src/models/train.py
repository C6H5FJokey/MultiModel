import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.data.data_preprocessing import CustomDataset1
from src.models.model import Type2Model
import os
import time

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), '../../logs'))

dataset = CustomDataset1(3)
batch_size = 8

# 设置数据集大小
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% 用于训练
val_size = int(0.1 * total_size)    # 10% 用于验证
test_size = total_size - train_size - val_size  # 剩余 10% 用于测试

# 划分数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


class DataAgent:
    def __init__(self, num_blocks=12, batch_size=8):
        self.dataset = CustomDataset1(num_blocks)
        self.total_size = total_size = len(self.dataset)
        self.train_size = train_size = int(0.8 * total_size)  # 80% 用于训练
        self.val_size = val_size = int(0.1 * total_size)    # 10% 用于验证
        self.test_size = total_size - train_size - val_size  # 剩余 10% 用于测试
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
    
    def get_train_loader():
        


def log_dice_loss_with_logit(y_hat, y, ep=1e-8):
    ce_loss = nn.BCEWithLogitsLoss()
    ce_total_loss = ce_loss(y_hat, y)
    y_hat = torch.sigmoid(y_hat)
    intersection = torch.sum(y_hat * y)
    dice = (2. * intersection + ep) / (torch.sum(y_hat) + torch.sum(y) + ep)
    dice = torch.clamp(dice, min=ep, max=1.0)
    dice_ce_loss = -1 * torch.log(dice) + ce_total_loss
    return dice_ce_loss

def train(net, device, loss_fn=log_dice_loss_with_logit, offset=94, num_epochs=200):

    # 初始化模型
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 训练循环
    timer_tik = time.time()
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            labels = labels[:, :, offset:-offset, offset:-offset].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            l = loss_fn(outputs, labels)
            l.backward()
            optimizer.step()
            running_loss += l.item()
            if (i % (len(train_loader) // 10) == 0): print(f'epoch: {epoch+1} / {num_epochs}, batch: {i+1} / {len(train_loader)}, loss:{l}')
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
                labels = labels[:, :, offset:-offset, offset:-offset].to(device)
                outputs = net(inputs)
                l = loss_fn(outputs, labels)
                val_loss += l.item()

        print(f'Epoch {epoch+1}, train_loss: {running_loss/len(train_loader)}, val_loss: {val_loss/len(val_loader)}, timer: {time.time() - timer_tik}')
        writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch)
    # 保存模型
    print(f'train end, timer: {time.time() - timer_tik}')
    save_model_incrementally(net)


def save_model_incrementally(net, directory=os.path.dirname(__file__), base_name='net'):
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


if __name__ == "__main__":
    net = Type2Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    train(net, device, num_epochs=10)
