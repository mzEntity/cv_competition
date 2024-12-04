from PIL import Image
from torchvision import transforms
import random
import numpy as np
import os
# 自定义数据增强类
class RandomImageEnhance:
    def __init__(self):
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.rotate = transforms.RandomRotation(degrees=[-45,45])
        self.brightness = transforms.ColorJitter(brightness=[0.6,0.9])
        self.contrast = transforms.ColorJitter(contrast=[0.6,0.9])
        self.saturation = transforms.ColorJitter(saturation=[0.6,0.9])
        self.hue = transforms.ColorJitter(hue=[-0.3,0.3])
        
    def __call__(self, img):
        # 随机水平翻转
        img = self.flip(img)
        
        # 随机旋转
        img = self.rotate(img)
        
        # 随机亮度、对比度、饱和度、色调增强
        img = self.brightness(img)
        img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        
        # 随机添加高斯噪声
        if random.random() < 0.5:
            mean = random.uniform(-5, 5)
            std = random.uniform(10, 40)
            img = self.add_gaussian_noise(img,mean,std)
        
        return img

    def add_gaussian_noise(self, img, mean=0, std=25):
        """
        向图像添加高斯噪声
        :param img: 输入图像
        :param mean: 高斯噪声的均值
        :param std: 高斯噪声的标准差
        :return: 添加噪声后的图像
        """
        # 将PIL图像转换为numpy数组
        img_np = np.array(img)
        
        # 生成高斯噪声
        noise = np.random.normal(mean, std, img_np.shape)
        
        # 添加噪声并确保像素值在0-255之间
        noisy_img = img_np + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        
        # 将噪声图像转换回PIL图像
        noisy_img = Image.fromarray(np.uint8(noisy_img))
        
        return noisy_img

# 保存增强图像并更新标签文件
def save_augmented_images_and_labels(image, label, file_name, save_dir, label_file):
    #os.makedirs(save_dir, exist_ok=True)
    augmented_images = []

    # 数据增强器
    aug = RandomImageEnhance()

    # 随机设定每个原数据生成的增强数据的数量
    augmented_images_number = random.randint(1, 2)

    for i in range(augmented_images_number):
        augmented_image = aug(image)
        augmented_images.append(augmented_image)

    # 保存增强后的图像并更新标签
    for i, aug_image in enumerate(augmented_images):
        # 构造增强后的图像文件名
        new_file_name = f"{os.path.splitext(file_name)[0]}_aug{i+1}.jpg"
        new_file_path = os.path.join(save_dir, new_file_name)

        # 保存图片
        aug_image.save(new_file_path)
        
        # 更新标签文件
        with open(label_file, "a") as f:
            f.write(f"{new_file_name}\t{label}\n")

# 主程序
def main():
    # 原始图片和标签的路径
    image_dir = r"C:/Users/xyz/Desktop/CVLab3/trainset/trainset"  # 原始图片文件夹
    label_file = r"C:/Users/xyz/Desktop/CVLab3/annotations/train.txt"  # 原始标签文件
    new_label_file = r"C:/Users/xyz/Desktop/CVLab3/annotations/augmented.txt" # 增强数据标签文件
    save_dir = r"C:/Users/xyz/Desktop/CVLab3/trainset/augmented_images" # 增强后的图片保存目录

    # 读取标签文件
    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        # 解析图片文件名和标签
        file_name = parts[0].replace("*", "_")  # 动态将 * 替换为 _
        label = int(parts[1])
        image_path = os.path.join(image_dir, file_name)

        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            continue

        # 读取图像并确保是 RGB 格式
        image = Image.open(image_path)
        # 增强图像并更新标签
        save_augmented_images_and_labels(image, label, file_name, save_dir, new_label_file)

    print(f"所有增强图像及标签已保存到 {save_dir}，标签已更新到 {new_label_file}")

if __name__ == "__main__":
    main()
