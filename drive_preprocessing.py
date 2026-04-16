import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DRIVEDataset(Dataset):
    """DRIVE数据集处理类"""
    
    def __init__(self, data_dir, split='train', patch_size=48, stride=16, 
                 augment=True, normalize=True):
        """
        Args:
            data_dir: DRIVE数据集根目录
            split: 'train', 'test', or 'all'
            patch_size: 提取patch的大小
            stride: patch提取步长
            augment: 是否进行数据增强
            normalize: 是否标准化
        """
        self.data_dir = data_dir
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.normalize = normalize
        
        self.images = []
        self.masks = []
        self.patches = []
        
        self._load_data()
        self._extract_patches()
        
    def _load_data(self):
        """加载DRIVE数据集"""
        if self.split in ['train', 'all']:
            train_img_dir = os.path.join(self.data_dir, 'train', 'image')
            train_mask_dir = os.path.join(self.data_dir, 'train', 'label')
            
            if os.path.exists(train_img_dir):
                for img_name in sorted(os.listdir(train_img_dir)):
                    if img_name.endswith('.tif'):
                        img_path = os.path.join(train_img_dir, img_name)
                        # 根据实际文件名格式生成对应的mask文件名
                        mask_name = img_name.replace('.tif', '.png')
                        mask_path = os.path.join(train_mask_dir, mask_name)
                        
                        if os.path.exists(mask_path):
                            self.images.append(img_path)
                            self.masks.append(mask_path)
        
        if self.split in ['test', 'all']:
            test_img_dir = os.path.join(self.data_dir, 'test', 'image')
            test_mask_dir = os.path.join(self.data_dir, 'test', 'label')
            
            if os.path.exists(test_img_dir):
                for img_name in sorted(os.listdir(test_img_dir)):
                    if img_name.endswith('.tif'):
                        img_path = os.path.join(test_img_dir, img_name)
                        # 根据实际文件名格式生成对应的mask文件名
                        mask_name = img_name.replace('.tif', '.png')
                        mask_path = os.path.join(test_mask_dir, mask_name)
                        
                        if os.path.exists(mask_path):
                            self.images.append(img_path)
                            self.masks.append(mask_path)
        
        print(f"Loaded {len(self.images)} images for {self.split} split")
    
    def _extract_patches(self):
        """从图像中提取patches"""
        self.patches = []
        
        for img_path, mask_path in zip(self.images, self.masks):
            # 读取图像和mask
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                continue
            
            # 转换为绿色通道（对血管分割最有效）
            if len(image.shape) == 3:
                image = image[:, :, 1]  # 绿色通道
            
            # 归一化到[0,1]
            image = image.astype(np.float32) / 255.0
            mask = (mask > 128).astype(np.float32)  # 二值化mask
            
            h, w = image.shape
            
            # 滑动窗口提取patches
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # 过滤全黑或接近全黑的patches
                    if np.mean(img_patch) > 0.1:
                        self.patches.append({
                            'image': img_patch,
                            'mask': mask_patch,
                            'source': os.path.basename(img_path)
                        })
        
        print(f"Extracted {len(self.patches)} patches")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        image = patch['image'].copy()  # 确保数组连续性
        mask = patch['mask'].copy()    # 确保数组连续性
        
        # 数据增强
        if self.augment and self.split == 'train':
            image, mask = self._apply_augmentation(image, mask)
        
        # 标准化
        if self.normalize:
            image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        # 转换为tensor
        image = torch.from_numpy(image).unsqueeze(0).float()  # [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float()    # [1, H, W]
        
        return image, mask
    
    def _apply_augmentation(self, image, mask):
        """应用数据增强"""
        # 随机旋转
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy()  # 添加.copy()确保数组连续性
            mask = np.rot90(mask, k).copy()    # 添加.copy()确保数组连续性
        
        # 随机翻转
        if random.random() > 0.5:
            image = np.fliplr(image).copy()  # 添加.copy()确保数组连续性
            mask = np.fliplr(mask).copy()    # 添加.copy()确保数组连续性
        
        if random.random() > 0.5:
            image = np.flipud(image).copy()  # 添加.copy()确保数组连续性
            mask = np.flipud(mask).copy()    # 添加.copy()确保数组连续性
        
        # 随机亮度调整
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)
        
        # 随机对比度调整
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            image = np.clip((image - 0.5) * alpha + 0.5, 0, 1)
        
        return image, mask

class DRIVETestDataset(Dataset):
    """DRIVE测试数据集，用于完整图像预测"""
    
    def __init__(self, data_dir, normalize=True):
        self.data_dir = data_dir
        self.normalize = normalize
        
        self.images = []
        self.masks = []
        self.image_names = []
        
        self._load_test_data()
    
    def _load_test_data(self):
        """加载测试数据"""
        test_img_dir = os.path.join(self.data_dir, 'test', 'image')
        test_mask_dir = os.path.join(self.data_dir, 'test', 'label')
        
        for img_name in sorted(os.listdir(test_img_dir)):
            if img_name.endswith('.tif'):
                img_path = os.path.join(test_img_dir, img_name)
                mask_name = img_name.replace('.tif', '.png')
                mask_path = os.path.join(test_mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)
                    self.image_names.append(img_name)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 读取完整图像
        image = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        
        # 转换为绿色通道
        if len(image.shape) == 3:
            image = image[:, :, 1]
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        mask = (mask > 128).astype(np.float32)
        
        if self.normalize:
            image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        
        # 转换为tensor
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask, self.image_names[idx]

def get_data_loaders(data_dir, batch_size=16, patch_size=48, stride=16, num_workers=4):
    """获取训练和验证数据加载器"""
    
    # 训练数据集
    train_dataset = DRIVEDataset(
        data_dir=data_dir,
        split='train',
        patch_size=patch_size,
        stride=stride,
        augment=True,
        normalize=True
    )
    
    # 从训练数据中分割出验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_test_loader(data_dir, batch_size=1, num_workers=4):
    """获取测试数据加载器"""
    test_dataset = DRIVETestDataset(data_dir, normalize=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

def predict_full_image(model, image, patch_size=48, stride=16, device='cuda'):
    """对完整图像进行patch-based预测"""
    model.eval()
    
    h, w = image.shape[2], image.shape[3]  # [1, 1, H, W]
    
    # 创建预测结果画布
    prediction = torch.zeros_like(image)
    count_map = torch.zeros_like(image)
    
    with torch.no_grad():
        # 滑动窗口预测
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # 提取patch
                patch = image[:, :, y:y+patch_size, x:x+patch_size].to(device)
                
                # 预测
                patch_pred = model(patch).cpu()
                
                # 累积结果
                prediction[:, :, y:y+patch_size, x:x+patch_size] += patch_pred
                count_map[:, :, y:y+patch_size, x:x+patch_size] += 1
    
    # 平均化重叠区域
    prediction = prediction / (count_map + 1e-8)
    
    return prediction

def visualize_sample(dataset, num_samples=4):
    """可视化数据集样本"""
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]
        
        # 显示图像
        axes[0, i].imshow(image.squeeze(), cmap='gray')
        axes[0, i].set_title(f'Image {idx}')
        axes[0, i].axis('off')
        
        # 显示mask
        axes[1, i].imshow(mask.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mask {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()