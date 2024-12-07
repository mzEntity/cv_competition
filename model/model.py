import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入通道为3 (RGB)，输出通道为32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 下采样一半
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 卷积层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # 再次下采样
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                         # 展平
            nn.Linear(64 * 56 * 56, 128),                        # 全连接层
            nn.ReLU(),
            nn.Linear(128, 1)                                    # 最终输出单个值（预测年龄）
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    
class MyResNetModel(nn.Module):
    def __init__(self):
        super(MyResNetModel, self).__init__()
        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    
class MyResNetModel2(nn.Module):
    def __init__(self):
        super(MyResNetModel2, self).__init__()
        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False
            
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
