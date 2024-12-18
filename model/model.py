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
    
class DualModel(nn.Module):
    def __init__(self, body_model_path, face_model_path):
        super(DualModel, self).__init__()
        
        self.body = MyResNetModel2()
        self.face = MyResNetModel2()
        
        self.body.load_state_dict(torch.load(body_model_path))
        self.face.load_state_dict(torch.load(face_model_path))
        
        
    def forward(self, is_face, x):
        if is_face:
            return self.face(x)
        else:
            return self.body(x)
    
    
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
        # 将最后的分类层换成回归层
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MyEfficientNetModel(nn.Module):
    def __init__(self):
        super(MyEfficientNetModel, self).__init__()

        efficientnet = models.efficientnet_b0(pretrained=True)
        for param in efficientnet.parameters():
            param.requires_grad = False

        self.efficientnet = nn.Sequential(*list(efficientnet.children())[:-1])
        self.fc = nn.Linear(efficientnet.classifier[1].in_features, 1)

    def forward(self, x):
        x = self.efficientnet(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    
class MyVGGModel(nn.Module):
    def __init__(self):
        super(MyVGGModel, self).__init__()
        vgg = models.vgg16(pretrained=True)

        self.vgg = nn.Sequential(*list(vgg.features.children()))
        
        for param in self.vgg.parameters():
            param.requires_grad = False

        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)  # 示例输入
            sample_output = self.vgg(sample_input)
            flat_size = sample_output.view(sample_output.size(0), -1).size(1)

        self.fc = nn.Linear(flat_size, 1)

    def forward(self, x):
        x = self.vgg(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
