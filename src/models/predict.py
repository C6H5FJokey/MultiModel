import torch
from src.utils import log_dice_loss_with_logit, bin_dice_eval_with_logit, DataAgent
from src.models.model import Type2Model
import time


def predict(net, device, test_loader, loss_fn=log_dice_loss_with_logit):
    net.eval()
    outputs_l = []
    labels_l = []
    loss_l = []
    dice_l = []
    timer_tik = time.time()
    offset= 94 if not net.use_padding else 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            if offset > 0:
                labels = labels[:, :, offset:-offset, offset:-offset]
            labels = labels.to(device)
            labels_l.append(labels)
            outputs = net(inputs)
            outputs_l.append(outputs.detach())
            l = loss_fn(outputs, labels)
            loss_l.append(l)
            dice_l.append(bin_dice_eval_with_logit(outputs, labels))
        loss_l = torch.tensor(loss_l)
        print(f'avg_loss: {loss_l.mean()}, avg_dice: {torch.cat(dice_l).mean()}, timer: {time.time() - timer_tik} ')
        return (outputs_l, labels_l, loss_l, dice_l)
    


if __name__ == '__main__':    
    if 'da' not in locals():
        da = DataAgent(3)
    if 'device' not in locals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'net' not in locals():
        net = Type2Model().to(device)
    predict_result = predict(net, device, da.get_test_loader())
    