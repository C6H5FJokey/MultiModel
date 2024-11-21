import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.models.model import Type2Model
import os
import time
from src.utils import DataAgent, set_seed, save_model_incrementally, log_dice_loss_with_logit

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), '../../logs'))


def train(net, device, train_loader, val_loader, loss_fn=log_dice_loss_with_logit, num_epochs=200):

    # 初始化模型
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    offset= 94 if not net.use_padding else 0
    # 训练循环
    timer_tik = time.time()
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            if offset > 0:
                labels = labels[:, :, offset:-offset, offset:-offset]
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            l = loss_fn(outputs, labels)
            l.backward()
            optimizer.step()
            running_loss += l.item()
            print(f'epoch: {epoch+1} / {num_epochs}, batch: {i+1} / {len(train_loader)}, loss: {l} ', end='\r')
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
                if offset > 0:
                    labels = labels[:, :, offset:-offset, offset:-offset]
                labels = labels.to(device)
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
    net = Type2Model(use_padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    if 'da' not in locals():
        da = DataAgent(3)
    train(net, device, da.get_train_loader(), da.get_val_loader(), num_epochs=10)
