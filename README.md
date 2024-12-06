# 多模态肿瘤语义分割

## 项目简介

本项目旨在使用深度学习技术对 CT、PET和MRI 图像进行处理和分析。我实现了一个自定义的神经网络模型，用于对肿瘤图像进行语义分割。该模型能够有效处理 MRI 图像。

## 功能特性

- 图像预处理和标准化
- 自定义深度学习模型实现
- 训练过程中的 checkpoint 保存和恢复
- 模型评估和性能分析

## 环境要求

- matplotlib==3.8.4
- numpy==2.1.3
- opencv_python==4.10.0.84
- pandas==2.2.3
- pydicom==3.0.1
- SimpleITK==2.4.0
- SimpleITK==2.4.0
- torch==2.4.1

## 安装指南

1. 克隆仓库：

```
git clone git@github.com:C6H5FJokey/multiModel.git
cd multiModel
```

2. 创建并激活虚拟环境（可选但推荐）：

```
python -m venv venv
source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
```

3. 安装依赖项：

```
pip install -r requirements.txt
```


## 使用说明

1. 数据准备：

从 https://www.cancerimagingarchive.net/collection/soft-tissue-sarcoma 下载图像数据，并放置在 `data/raw` 目录下。

2. 数据预处理：

在项目下运行python交互模式
```
from src.data.data_preprocessing import spawn_datasets_and_labels, split_batch
spawn_datasets_and_labels() # 自动处理数据，处理后的样本数据位于 data/processed/datasets，标注位于 data/processed/labels
split_batch() # 将数据进行切分，处理后的数据位于 data/processed/datasets
```

3. 训练数据：

直接使用脚本（不推荐）
```
python src/models/train.py
```
或进入交互模式，手动执行
```
import torch
from src.models.model import Type2Model
from src.models.train import train
from src.utils import DataAgent
net = Type2Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
train(net, device, da.get_train_loader(6), da.get_val_loader(6), num_epochs=200, 
          resume_from_checkpoint=None, save_interval=10, keep_n_checkpoints=5)
```
其中da的get_xxx_loader()可以设置训练批量，`resume_from_checkpoint` 设置 `best_model.pth.gz` 或 `checkpoint_epoch_{epoch_num}.pth.gz`可进行训练断点恢复

4. 验证数据：

进入交互模式，手动执行
```
import torch
from src.models.model import Type2Model
from src.models.train import predict
from src.utils import DataAgent
net = Type2Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
# 如果验证训练效果最好的模型
checkpoint = load_compressed_checkpoint('src/models/checkpoints', 'best_model.pth.gz')
net.load_state_dict(checkpoint['model_state_dict'])
# 如果验证最后训练完成的模型
net.load_state_dict(torch.load('net_{num}.pth')) # num为目录中具体的模型
predict_result = predict(net, device, da.get_test_loader())
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Reference

This project is an implementation of the methods described in:

Vallières M, Freeman C R, Skamene S R, et al. A radiomics model from joint FDG-PET and MRI texture features for the prediction of lung metastases in soft-tissue sarcomas of the extremities [J]. Physics in Medicine and Biology, 2015, 60(14): 5471-96.

Guo Z, Li X, Huang H, et al. Deep Learning-Based Image Segmentation on Multimodal Medical Imaging [J]. IEEE Transactions on Radiation and Plasma Medical Sciences, 2019, 3(2): 162-9.
