import torch
import torch.nn.functional as F

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