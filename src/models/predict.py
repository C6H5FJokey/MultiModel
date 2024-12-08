import torch
from src.utils import log_dice_loss_with_logit, bin_dice_eval_with_logit, DataAgent, get_model
from src.models.model import Type2Model
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, auc

def predict(net, device, test_loader, loss_fn=log_dice_loss_with_logit):
    net.eval()
    outputs_l = []
    labels_l = []
    loss_l = []
    dice_l = []
    timer_tik = time.time()
    offset= 94 if not get_model(net).use_padding else 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = tuple(input_tensor.to(device) for input_tensor in inputs)
            if offset > 0:
                if len(labels.shape)==4:
                    labels = labels[:, :, offset:-offset, offset:-offset]
                elif len(labels.shape)==3:
                    labels = labels[:, offset:-offset, offset:-offset]
            labels = labels.to(device)
            labels_l.append(labels)
            outputs = net(inputs)
            outputs_l.append(outputs.detach())
            l = loss_fn(outputs, labels)
            loss_l.append(l)
            dice_l.append(bin_dice_eval_with_logit(outputs, labels))
        loss_l = torch.tensor(loss_l)
        dice_l_ = torch.cat(dice_l)
        labels_l_ = torch.cat(labels_l)
        dice_bg = dice_l_[(labels_l_.sum(dim=(1,2))==0)].mean()
        dice_fg = dice_l_[(labels_l_.sum(dim=(1,2))!=0)].mean()
        print(f'avg_loss: {loss_l.mean()}, avg_dice: {torch.cat(dice_l).mean()}, avg_iou: {iou_score(outputs_l, labels_l).mean()}, dice_fg: {dice_fg}, dice_bg: {dice_bg}, timer: {time.time() - timer_tik} ')
        return (outputs_l, labels_l, loss_l, dice_l)
    

def box_plt(dice_l, pic_name=''):
    # 假设你有一个存储 Dice 结果的列表
    dice_l = torch.cat(dice_l)
    dice_scores = np.array(dice_l.cpu())
    # 使用 seaborn 绘制箱形图
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=dice_scores)

    # 添加标题和标签
    plt.title('Dice Scores Boxplot')
    plt.ylabel('Dice Score')
    
    if pic_name:
        # 保存图形为矢量图
        plt.savefig(os.path.join(os.path.dirname(__file__), '../../data/extra', pic_name+'.svg'))  # 或者使用 'dice_scores_boxplot.pdf'
        plt.close()
    else:
        plt.show()


def roc_plt(outputs_l, labels_l, pic_name=''):
    true_labels = np.array(torch.cat(labels_l).cpu()).flatten()  # 真实标签，展平的数组
    predicted_probs = np.array(torch.sigmoid(torch.cat(outputs_l)).cpu()).flatten()  # 预测概率，展平的数组
    
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    
    # 计算 AUC
    roc_auc = auc(fpr, tpr)
    
    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    if pic_name:
        plt.savefig(os.path.join(os.path.dirname(__file__), '../../data/extra', pic_name+'.svg'))
        plt.close()
    else:
        plt.show()


def iou_score(outputs_l, labels_l, threshold=0.5):
    # 二值化预测结果
    y_pred_bin = ((torch.sigmoid(torch.cat(outputs_l))) >= threshold).to(torch.float32)
    y_true = torch.cat(labels_l)
    
    if len(y_true.shape) == 3:
        y_true = y_true.unsqueeze(1)
    # 计算交集和并集
    intersection = (y_true * y_pred_bin).sum(dim=(1, 2, 3))
    union = y_true.sum(dim=(1, 2, 3)) + y_pred_bin.sum(dim=(1, 2, 3)) - intersection
    
    # 计算 IoU
    iou = torch.zeros_like(union)
    iou[union == 0] = (y_pred_bin.sum(dim=(1, 2, 3))[union == 0] == 0).to(torch.float32)
    iou[union != 0] = intersection[union != 0] / union[union != 0]
    return iou

if __name__ == '__main__':    
    if 'da' not in locals():
        da = DataAgent(3)
    if 'device' not in locals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if 'net' not in locals():
        net = Type2Model().to(device)
    predict_result = predict(net, device, da.get_test_loader())
    