import os
import pandas as pd
import pydicom
import numpy as np
import cv2
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

base_path = os.path.join(os.path.dirname(__file__), '../../data/raw')
base_save_path = os.path.join(os.path.dirname(__file__), '../../data/processed')


def load_metadata():
    file_path = os.path.join(base_path, 'metadata.csv')
    if not os.path.exists(file_path):
        return
    metadata = pd.read_csv(file_path)
    metadata = metadata[metadata['Study Description'].isin(
         ('PET CT with registered MR', 
          'PET CT', 
          'PETCT with added MR', 
          'CT PET with registered MR'
          )
         )]
    
    metadata.loc[metadata['Series Description'].isin(
        ('RTstructAlignedT1toPET',
         'AlignedT1toPETBOX',
         'AllignedT1toPETBOX',
         'RTstructAlignedTtoPET',
         'RTstructAlignedT1toPETBOX'
         )
        ), 'type'] = 'T1'
    metadata.loc[metadata['Series Description'].isin(
        ('RTstructAlignedT2FStoPET',
         'AlignedT2FStoPETBOX',
         'RTstructAlignedSTIRtoPET',
         'AlignedSTIRtoPETBOX',
         'RTstructAlignedSTIRtoPETBOX'
         )
        ), 'type'] = 'T2'
    metadata.loc[metadata['Series Description'].isin(
        ('RTstructPET',
         'PET AC',
         'WB2DAC',
         'PED2DAC',
         'PET AC 21',
         'LEGS2DAC',
         'RTStructPET'
         )
        ), 'type'] = 'PET'
    metadata.loc[metadata['Series Description'].isin(
        ('RTstructCT',
         'CT IMAGES - RESEARCH',
         'StandardFull',
         'CT IMAGES - LEGS - RESEARCH',
         'StandardFull - RESEARCH',
         'CT IMAGES - ARMS - RESEARCH',
         'StandardFull - CT LEGS - RESEARCH',
         'CT IMAGES - BODY RESEARCH',
         'CT IMAGES - BODY - RESEARCH',
         'RTStructCT',
         'CT IMAGES'
         )
        ), 'type'] = 'CT'
    
    return metadata

class ImageAgent:
    metadata = None
    
    def _load_metadata():
        ImageAgent.metadata = load_metadata()
    
    def __init__(self, **kwargs):
        if not metadata:
            ImageAgent._load_metadata()
        self._volumes_points = []
        self._img_data = None
        self._st_data = None
        self._sts = None
        self._idx = None
        self._type_name = None
        self._mask = []
        self.z_min = None
        self.z_max = None
        self._z0 = {}
        self._z1 = {}
        sts = None
        idx = None
        type_name = None
        if 'sts' in kwargs:
            sts = self._sts = kwargs.get('sts')
        if 'idx' in kwargs:
            idx = self._idx = kwargs.get('idx')
        if 'type_name' in kwargs:
            type_name = self._type_name = kwargs.get('type_name')
        if sts and type_name:
            self._st_data = pydicom.dcmread(get_st_path(sts, type_name, metadata=metadata))
            self._volumes_points = get_volumes_points(self._st_data)
            self.z_min, self.z_max = z_scoop(self._volumes_points)
            if idx != None:
                self._img_data = pydicom.dcmread(get_image_path(sts, type_name, idx))
                self._mask = [
                    get_mask(self._img_data, self._volumes_points[0]),
                    get_mask(self._img_data, self._volumes_points[1])
                    ]

    @property
    def mask(self, idx=-1):
        if idx == -1:
            return self._mask
        return self._mask[idx]
    
    
    @property
    def sts(self):
        return self._sts
    
    
    @sts.setter
    def sts(self, value):
        sts = value
        if self.img_data and self.sts != sts:
            self._z0 = {}
            self._z1 = {}
        if self._type_name:
            self._st_data = pydicom.dcmread(get_st_path(sts, self._type_name, metadata=metadata))
            self._volumes_points = get_volumes_points(self._st_data)
            self.z_min, self.z_max = z_scoop(self._volumes_points)
        self._idx = None
        self._img_data = None
        self._sts = value
        return self._st_data
    
    
    @property
    def type_name(self):
        return self._type_name
    
    
    @type_name.setter
    def type_name(self, value):
        type_name = self._type_name = value
        if self._sts:
            self._st_data = pydicom.dcmread(get_st_path(self._sts, type_name, metadata=metadata))
            self._volumes_points = get_volumes_points(self._st_data)
            self.z_min, self.z_max = z_scoop(self._volumes_points)
        if self._img_data:
            zt = self._img_data.ImagePositionPatient[2]
            group = metadata[metadata['Subject ID'] == f'STS_{self._sts:03d}']
            img_group = group[group['Modality'] != 'RTSTRUCT']
            folder = img_group.loc[img_group['type'] == type_name, 'File Location'].values[0]
            num = len(os.listdir(os.path.join(base_path, folder)))
            if type_name in self._z0 and type_name in self._z1:
                z0 = self._z0.get(type_name)
                z1 = self._z1.get(type_name)
            else:
                z0 = pydicom.dcmread(os.path.join(base_path, folder, f'1-{1:0{len(str(num))}d}.dcm')).ImagePositionPatient[2]
                z1 = pydicom.dcmread(os.path.join(base_path, folder, f'1-{2:0{len(str(num))}d}.dcm')).ImagePositionPatient[2]
                self._z0[type_name] = z0
                self._z1[type_name] = z1
            self._idx = int(round((zt - z0)/(z1 - z0)))
            self._img_data = pydicom.dcmread(os.path.join(base_path, folder, f'1-{self._idx+1:0{len(str(num))}d}.dcm'))
            self._mask = [
                get_mask(self._img_data, self._volumes_points[0]),
                get_mask(self._img_data, self._volumes_points[1])
                ]
        else:    
            self._idx = None
            self._img_data = None
            self._mask = []
        return self._st_data
    
    
    @property
    def idx(self):
        return self._idx
    
    
    @idx.setter
    def idx(self, value):
        idx = self._idx = value
        if self._sts and self._type_name:
            self._img_data = pydicom.dcmread(get_image_path(self._sts, self._type_name, idx, metadata=metadata))
            self._mask = [
                get_mask(self._img_data, self._volumes_points[0]),
                get_mask(self._img_data, self._volumes_points[1])
                ]
        return self._img_data
    
    
    @property
    def st_data(self):
        return self._st_data
    
    
    @property
    def img_data(self):
        return self._img_data


    @property
    def pos(self):
        if self._img_data:
            return self._img_data.ImagePositionPatient
        return None
    

    def __len__(self):
        if self._st_data:
            folder = get_folder_path(self._sts, self._type_name)
            return len(os.listdir(os.path.join(base_path, folder)))
        return 0
            
            

    def plt_raw_and_mask(self):
        
        plt.figure(figsize=(15, 5))
        # 原始图像
        plt.subplot(1, 3, 1)
        plt.title(f'Original Image - {self._type_name}')
        plt.imshow(self._img_data.pixel_array, cmap='gray')
        
        # 分割掩码
        plt.subplot(1, 3, 2)
        plt.title('Segmentation Mask')
        plt.imshow(self._mask[0], cmap='gray')
        
        # 叠加轮廓
        plt.subplot(1, 3, 3)
        plt.title('Overlay with Contour')
        plt.imshow(self._img_data.pixel_array, cmap='gray')
        plt.contour(self._mask[0], colors='r', linewidths=1)
    
    
    @property
    def raw_and_mask(self):
        self.plt_raw_and_mask()


    @property
    def shape(self):
        if self._img_data:
            return (self._img_data.Rows, self._img_data.Columns)
        return None
        
        

def groupby_metadata():
    metadata = load_metadata()
    return metadata.groupby('Subject ID')

def get_physical_info(image_file):
    return (np.array(image_file.ImagePositionPatient),
            np.array(image_file.ImageOrientationPatient).reshape(2, 3),
            image_file.PixelSpacing,
            image_file.pixel_array.shape)

def extract_slice_from_volume(volume_points, z_coord, slice_thickness):
    """
    从三维体中提取沿 z 轴的切片。

    :param volume_points: 三维体的轮廓点集，大小为 (n, 3)
    :param z_coord: 切片的 z 坐标
    :param slice_thickness: 切片的厚度
    :return: 二维切片 (x, y)
    """
    # 筛选出在指定 z 坐标附近的点
    mask = (volume_points[:, 2] >= z_coord - slice_thickness / 2) & (volume_points[:, 2] <= z_coord + slice_thickness / 2)
    
    # 提取切片上的点
    slice_points = volume_points[mask]

    # 提取 x, y 坐标
    slice_2d = slice_points[:, :2]

    return slice_2d

def get_volumes_points(rtstruct_file):
    """
    获得RTstruct信息中的三维边缘信息。
    通常信息不止一组，请根据RTstruct里StructureSetROISequence中的信息进行区分。

    Parameters
    ----------
    rtstruct_file : dataset.FileDataset
        由pydicom读取的dcm文件。
    Returns
    -------
    volumes_points : list
        获得三维边缘信息，可能有多组。

    """
    volumes_points = []
    for roi_contour in rtstruct_file.ROIContourSequence:
        points = []
        for contour_sequence in roi_contour.ContourSequence:
            contour_data = contour_sequence.ContourData
            points.append(np.array(contour_data).reshape(-1, 3))
        volumes_points.append(np.concatenate(points))
    return volumes_points
    

def get_mask(dicom_data, volume_points, sitk_data=None):
    """
    根据三维边缘信息生成掩码。

    Parameters
    ----------
    dicom_data : dataset.FileDataset
        pydicom导入图像。
    volume_points : list
        三维边缘点信息。
    sitk_data : SimpleITK.Image
        SimpleITK.Image图像
    Returns
    -------
    mask : np.ndarray
        二维掩码。
    """
    slice_thickness = dicom_data.SliceThickness
    image_pos = sitk_data.GetOrigin() if sitk_data else np.array(dicom_data.ImagePositionPatient)
    z_coord = image_pos[2]
    points = extract_slice_from_volume(volume_points, z_coord, slice_thickness) - image_pos[:2]
    points = np.round(points).astype(np.int32)
    mask = np.zeros_like(sitk.GetArrayFromImage(sitk_data).squeeze() if sitk_data else dicom_data.pixel_array)
    if points.size > 0:
        cv2.fillPoly(mask, [points], 1)
    return mask


def resample_image(image, target_spacing, target_origin, target_size, target_direction):
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(target_size)
    resample.SetOutputDirection(target_direction)
    resample.SetOutputOrigin(target_origin)
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(image)


def get_folder_path(sts, type_name, metadata=load_metadata()):
    group = metadata[metadata['Subject ID'] == f'STS_{sts:03d}']
    img_group = group[group['Modality'] != 'RTSTRUCT']
    folder = img_group.loc[img_group['type'] == type_name, 'File Location'].values[0]
    return os.path.join(base_path, folder)


def get_image_path(sts, type_name, idx, metadata=load_metadata()):
    group = metadata[metadata['Subject ID'] == f'STS_{sts:03d}']
    img_group = group[group['Modality'] != 'RTSTRUCT']
    folder = img_group.loc[img_group['type'] == type_name, 'File Location'].values[0]
    num = len(os.listdir(os.path.join(base_path, folder)))
    if idx >= num:
        raise IndexError('idx out of range')
    return os.path.join(base_path, folder, f'1-{idx+1:0{len(str(num))}d}.dcm')


def get_st_path(sts, type_name, metadata=load_metadata()):
    group = metadata[metadata['Subject ID'] == f'STS_{sts:03d}']
    st_group = group[group['Modality'] == 'RTSTRUCT']
    folder = st_group.loc[st_group['type'] == type_name, 'File Location'].values[0]
    return os.path.join(base_path, folder, '1-1.dcm')


def z_scoop(data):
    if type(data) == str:
        data = pydicom.dcmread(data)
    if type(data) == pydicom.dataset.FileDataset:
        data = get_volumes_points(data)
    if type(data) == list:
        data = data[0]
    return data[:, 2].min(), data[:, 2].max()



def spawn_datasets_and_labels(mask_type=None):
    metadata_groups = groupby_metadata()
    for name, group in metadata_groups:
        print(f"组名: {name}")
        st_group = group[group['Modality'] == 'RTSTRUCT']
        img_group = group[group['Modality'] != 'RTSTRUCT']
        if not mask_type:
            mask_type = 'T2'
        folder = st_group.loc[st_group['type'] == mask_type, 'File Location'].values[0]
        volumes_points = []
        for filename in os.listdir(os.path.join(base_path, folder)):
            if filename.endswith(".dcm"):
                file_path = os.path.join(base_path, folder, filename)
                dicom_data = pydicom.dcmread(file_path)
                volumes_points = get_volumes_points(dicom_data)
                break
        min_files = float('inf')
        min_type = ''
        for index, row in img_group.loc[:, ['type', 'File Location']].iterrows():
            num_files = len(os.listdir(os.path.join(base_path, row['File Location'])))
            if num_files < min_files:
                min_files = num_files
                min_type = row['type']
        folder = img_group.loc[img_group['type'] == min_type, 'File Location'].values[0]
        z_coords = []
        for filename in os.listdir(os.path.join(base_path, folder)):
            if filename.endswith('.dcm'):
                file_path = os.path.join(base_path, folder, filename)
                dicom_data = pydicom.dcmread(file_path)
                z_coords.append(dicom_data.ImagePositionPatient[2])
                sitk_data = sitk.ReadImage(file_path)
        z_min = min(z_coords)
        z_max = max(z_coords)
        z_coords = np.array(z_coords)
        targets = {}
        folder = img_group.loc[img_group['type'] == 'CT', 'File Location'].values[0]
        masks = [None] * len(z_coords)
        datasets = [None] * len(z_coords)
        for filename in os.listdir(os.path.join(base_path, folder)):
            if filename.endswith('.dcm'):
                file_path = os.path.join(base_path, folder, filename)
                dicom_data = pydicom.dcmread(file_path)
                z_coord = dicom_data.ImagePositionPatient[2]
                if z_coord < z_min - 1 or z_coord > z_max + 1: continue
                z_index = np.abs(z_coords - z_coord).argmin()
                z_coord = z_coords[z_index]
                mask = get_mask(dicom_data, volumes_points[0])
                masks[z_index] = mask
                # np.save(os.path.join(base_save_path, 'labels', f'{name}.npy'), mask)
                sitk_data = sitk.ReadImage(file_path)
                targets[z_index] = (
                    sitk_data.GetSpacing(),
                    sitk_data.GetOrigin(),
                    sitk_data.GetSize(),
                    sitk_data.GetDirection()
                    )
                np.save(os.path.join(base_save_path, 'datasets', f'{name}_CT.npy'), dicom_data.pixel_array)
                datasets[z_index] = dicom_data.pixel_array
        datasets = np.stack(datasets)
        masks = np.stack(masks)
        np.save(os.path.join(base_save_path, 'labels', f'{name}.npy'), masks)
        np.save(os.path.join(base_save_path, 'datasets', f'{name}_CT.npy'), datasets)
        for type_name in ['PET', 'T1', 'T2']:
            folder = img_group.loc[img_group['type'] == type_name, 'File Location'].values[0]
            for filename in os.listdir(os.path.join(base_path, folder)):
                if filename.endswith('.dcm'):
                    file_path = os.path.join(base_path, folder, filename)
                    dicom_data = pydicom.dcmread(file_path)
                    z_coord = dicom_data.ImagePositionPatient[2]
                    if z_coord < z_min or z_coord > z_max: continue
                    z_index = np.abs(z_coords - z_coord).argmin()
                    z_coord = z_coords[z_index]
                    sitk_data = sitk.ReadImage(file_path)
                    sitk_data = resample_image(sitk_data, *targets[z_index])
                    # np.save(os.path.join(base_save_path, 'datasets', f'{name}_{type_name}.npy'), sitk.GetArrayFromImage(sitk_data))
                    datasets[z_index] = sitk.GetArrayFromImage(sitk_data)
            datasets = np.stack(datasets)
            np.save(os.path.join(base_save_path, 'datasets', f'{name}_{type_name}.npy'), datasets)
        print("\n")
    

class CustomDataset1(Dataset):
    def __init__(self, num_blocks=12, normalize=True):
        self.normalize = normalize
        ct = []
        pet = []
        t1 = []
        t2 = []
        label = []
        for num in range(num_blocks):
            num = f"{num:02d}" 
            ct.append(torch.load(os.path.join(base_save_path, f'CT_{num}.pt'), weights_only=True))
            pet.append(torch.load(os.path.join(base_save_path, f'PET_{num}.pt'), weights_only=True))
            t1.append(torch.load(os.path.join(base_save_path, f'T1_{num}.pt'), weights_only=True))
            t2.append(torch.load(os.path.join(base_save_path, f'T2_{num}.pt'), weights_only=True))
            label.append(torch.load(os.path.join(base_save_path, f'label_{num}.pt'), weights_only=True))
        self.ct = torch.cat(ct)
        self.pet = torch.cat(pet)
        self.t1 = torch.cat(t1)
        self.t2 = torch.cat(t2)
        self.label = torch.cat(label)
        if self.normalize:
            MIN_BOUND = 150
            MAX_BOUND = 3000
            self.ct = (self.ct - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
            self.ct = torch.clamp(self.ct, min=0., max=1.0)
            self.pet = (self.pet - self.pet.min()) / (self.pet.max() - self.pet.min())
            self.t1 = (self.t1 - self.t1.min()) / (self.t1.max() - self.t1.min())
            self.t2 = (self.t2 - self.t2.min()) / (self.t2.max() - self.t2.min())
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        ct = self.ct[idx]
        pet = self.pet[idx]
        t1 = self.t1[idx]
        t2 = self.t2[idx]
        label = self.label[idx]

        return (ct, pet, t1, t2), label


    
class CustomDataset(Dataset):
    def __init__(self, normalize=True):
        num_blocks = 12
        data_dir = base_save_path
        self.ct_files = [os.path.join(data_dir, f'CT_{i:02d}.pt') for i in range(num_blocks)]
        self.pet_files = [os.path.join(data_dir, f'PET_{i:02d}.pt') for i in range(num_blocks)]
        self.t1_files = [os.path.join(data_dir, f'T1_{i:02d}.pt') for i in range(num_blocks)]
        self.t2_files = [os.path.join(data_dir, f'T2_{i:02d}.pt') for i in range(num_blocks)]
        self.label_files = [os.path.join(data_dir, f'label_{i:02d}.pt') for i in range(num_blocks)]
        self.normalize = normalize

    def __len__(self):
        return len(self.label_files) * 288  # 每个文件有 288 个样本
    
    def __getitem__(self, idx):
        block_idx = idx // 288
        sample_idx = idx % 288

        ct_block = torch.load(self.ct_files[block_idx], weights_only=True)
        pet_block = torch.load(self.pet_files[block_idx], weights_only=True)
        t1_block = torch.load(self.t1_files[block_idx], weights_only=True)
        t2_block = torch.load(self.t2_files[block_idx], weights_only=True)
        label_block = torch.load(self.label_files[block_idx], weights_only=True)
        
        if self.normalize:
            ct_block = (ct_block - ct_block.mean()) / ct_block.std()
            pet_block = (pet_block - pet_block.mean()) / pet_block.std()
            t1_block = (t1_block - t1_block.mean()) / t1_block.std()
            t2_block = (t2_block - t2_block.mean()) / t2_block.std()

        return (ct_block[sample_idx], pet_block[sample_idx], t1_block[sample_idx], t2_block[sample_idx]), label_block[sample_idx]
    
def split_batch():
    batch_num = 12
    for type_name in ['CT', 'PET', 'T1', 'T2']:
        dataset = []
        for num in range(1, 52):
            num = f"{num:03d}"
            dataset.append(np.load(os.path.join(base_save_path, 'datasets', f'STS_{num}_{type_name}.npy')))
        dataset = np.concatenate((dataset))
        batch_size = dataset.shape[0]//batch_num
        for i in range(batch_num):
            print(f'save: {type_name}_{i:02d}.pt batch_size: {batch_size}')
            torch.save(torch.from_numpy(dataset[i*batch_size:(i+1)*batch_size]).float().unsqueeze(1), os.path.join(base_save_path, f'{type_name}_{i:02d}.pt'))
            
    dataset = []
    for num in range(1, 52):
        num = f"{num:03d}"
        dataset.append(np.load(os.path.join(base_save_path, 'labels', f'STS_{num}.npy')))
    dataset = np.concatenate((dataset))
    batch_size = dataset.shape[0]//batch_num
    for i in range(batch_num):
        print(f'save : label_{i:02d}.pt batch_size: {batch_size}')
        torch.save(torch.from_numpy(dataset[i*batch_size:(i+1)*batch_size]).float().unsqueeze(1), os.path.join(base_save_path, f'label_{i:02d}.pt'))
    

if __name__ == '__main__':
    metadata = load_metadata()
    metadata_groups = groupby_metadata()
    print((metadata_groups.size() == 8).all())
    print(len(metadata_groups))
    # transform = transforms.Compose([
    #     transforms.Normalize(mean=[0.0], std=[1.0])  # 假设是单通道图像
    # ])
    # dataset = CustomDataset(transform=transform)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    