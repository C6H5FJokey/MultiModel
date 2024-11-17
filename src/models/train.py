import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.models.model import Type2Model
import os
import time
from src.utils import DataAgent, set_seed, save_model_incrementally

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), '../../logs'))


def log_dice_loss_with_logit(y_hat, y, ep=1e-8):
    ce_loss = nn.BCEWithLogitsLoss()
    ce_total_loss = ce_loss(y_hat, y)
    y_hat = torch.sigmoid(y_hat)
    intersection = torch.sum(y_hat * y)
    dice = (2. * intersection + ep) / (torch.sum(y_hat) + torch.sum(y) + ep)
    dice = torch.clamp(dice, min=ep, max=1.0)
    dice_ce_loss = -1 * torch.log(dice) + ce_total_loss
    return dice_ce_loss

def train(net, device, train_loader, val_loader, loss_fn=log_dice_loss_with_logit, offset=94, num_epochs=200):

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
            print(f'epoch: {epoch+1} / {num_epochs}, batch: {i+1} / {len(train_loader)}, loss:{l} ', end='\r')
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
    save_model_incrementally(net, os.path.join(os.path.dirname(__file__), '../../models'))


if __name__ == "__main__":
    set_seed(0)
    net = Type2Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    if 'da' not in locals():
        da = DataAgent(3)
    train(net, device, da.get_train_loader(), da.get_val_loader(), num_epochs=10)
