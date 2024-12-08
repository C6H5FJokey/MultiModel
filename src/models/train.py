import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.models.model import Type2Model, MultiResUNet
import os
import time
from src.utils import DataAgent, set_seed, save_model_incrementally, log_dice_loss_with_logit, load_compressed_checkpoint, save_compressed_checkpoint, custom_loss, get_model

writer = SummaryWriter(os.path.join(os.path.dirname(__file__), '../../logs'))


def train(net, device, train_loader, val_loader, loss_fn=log_dice_loss_with_logit, num_epochs=200, 
          resume_from_checkpoint=None, save_interval=10, keep_n_checkpoints=5):

    # 初始化模型
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    offset= 94 if not get_model(net).use_padding else 0
    
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from_checkpoint:
        checkpoint = load_compressed_checkpoint(os.path.join(os.path.dirname(__file__), 'checkpoints'), resume_from_checkpoint)
        get_model(net).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    # 训练循环
    timer_tik = time.time()
    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            if offset > 0:
                if len(labels.shape)==4:
                    labels = labels[:, :, offset:-offset, offset:-offset]
                elif len(labels.shape)==3:
                    labels = labels[:, offset:-offset, offset:-offset]
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            l = loss_fn(outputs, labels)
            l.backward()
            optimizer.step()
            running_loss += l.item()
            print(f'epoch: {epoch+1} / {num_epochs}, batch: {i+1} / {len(train_loader)}, loss: {l.item()} ', end='\r')
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
                if offset > 0:
                    if len(labels.shape)==4:
                        labels = labels[:, :, offset:-offset, offset:-offset]
                    elif len(labels.shape)==3:
                        labels = labels[:, offset:-offset, offset:-offset]
                labels = labels.to(device)
                outputs = net(inputs)
                l = loss_fn(outputs, labels)
                val_loss += l.item()
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}, train_loss: {avg_train_loss}, val_loss: {avg_val_loss}, timer: {time.time() - timer_tik}')
        writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch)
        
        # 保存最佳模型和定期checkpoint
        is_best = avg_val_loss < best_val_loss
        best_val_loss = min(avg_val_loss, best_val_loss)

        if is_best or (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': get_model(net).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }
            
            if is_best:
                save_compressed_checkpoint(checkpoint, os.path.join(os.path.dirname(__file__), 'checkpoints'), 'best_model.pth.gz')
            
            if (epoch + 1) % save_interval == 0:
                save_compressed_checkpoint(checkpoint, os.path.join(os.path.dirname(__file__), 'checkpoints'), f'checkpoint_epoch_{epoch+1}.pth.gz')
                
                checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
                # 保持最近N个checkpoints
                checkpoints = sorted(
                    [f for f in os.listdir(checkpoints_dir) if f.startswith('checkpoint_epoch_')],
                    key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x))
                )
                for old_checkpoint in checkpoints[:-keep_n_checkpoints]:
                    os.remove(os.path.join(os.path.dirname(__file__), 'checkpoints', old_checkpoint))
                
    # 保存模型
    print(f'train end, timer: {time.time() - timer_tik}')
    save_model_incrementally(net, os.path.join(os.path.dirname(__file__), '../../models'))


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"
    device_ids = [0, 1]
    set_seed(0)
    net = Type2Model(use_padding=True, unet=MultiResUNet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    if 'da' not in locals():
        da = DataAgent(1)
    train(net, device, da.get_train_loader(1), da.get_val_loader(1), num_epochs=10, loss_fn=custom_loss)
