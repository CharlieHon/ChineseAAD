import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, channel_num=32, kernel_size=17, class_num=2):
        super().__init__()

        # pad = nn.ZeroPad2d(((kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1, 0, 0))
        self.features = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=(channel_num, kernel_size)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(5, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, class_num)
        )

    def forward(self, x):
        # 因为输入的eeg数据形状是 (samples, channels)，即 (128, 32)
        # 升维 (4, 128, 32)   ->  (4, 1, 128, 32)
        x = x.unsqueeze(dim=1)
        # 转置(可选) (4, 1, 128, 32) -> (4, 1, 32, 128)
        # x = x.transpose(2, 3)
        # (4, 1, 32, 128)   ->      (4, 5, 1, 1)
        x = self.features(x)
        # (4, 5, 1, 1)      ->      (4, 5)
        x = torch.flatten(x, start_dim=1)
        # (4, 5)            ->      (4, 2)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = ConvNet()
    data = torch.randn(4, 32, 128)
    out = model(data)
    print(out.shape)
