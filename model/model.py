import torch.nn as nn

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
            nn.Linear(64 * 32 * 32, 128),                        # 全连接层
            nn.ReLU(),
            nn.Linear(128, 1)                                    # 最终输出单个值（预测年龄）
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x