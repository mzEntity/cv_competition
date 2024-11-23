import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        """
        Args:
            root_dir (string): 项目根目录。
            mode (string): 模式，包含train, validate, test
            transform (callable, optional): 可选的转换操作。
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.annotation_path = os.path.join(root_dir, f'data/annotations/{"val" if mode == "test" else "train"}.txt')
        self.img_path_dir = os.path.join(root_dir, f'data/{"val" if mode == "test" else "train"}set')
        
        self.samples = self.get_annotations(mode, self.annotation_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_path_dir, img_name).replace("*", "_")
        
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_annotations(self, mode, annotation_path):
        image_labels = []
        with open(annotation_path, 'r') as file:
            for line in file:
                img_name, label = line.strip().split()
                image_labels.append((img_name, int(label)))
                
        count = len(image_labels)
        if mode == "validate":
            image_labels = image_labels[:count//10]
        elif mode == "train":
            image_labels = image_labels[count//10:]
            
        return image_labels
    
def get_dataset(root_dir, mode):
    dataset = CustomDataset(root_dir, mode, None)
    return dataset
    
    
if __name__ == "__main__":
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 项目根目录
    root_dir = 'D://fudan//2024Autumn//CV//competition//cv_competition'

    # 创建数据集和数据加载器
    dataset = CustomDataset(root_dir=root_dir, train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 遍历数据并打印图像和标签的形状
    for images, labels in dataloader:
        print(f'Batch of images shape: {images.shape}')  # 打印图像的形状
        print(f'Batch of labels: {labels}')  # 打印标签
        break  # 只打印一个批次的数据以进行测试