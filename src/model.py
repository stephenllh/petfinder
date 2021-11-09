import torch
from torch import nn
import timm


class RegressorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=True, num_classes=0
        )
        dummy_input = torch.randn(2, 3, 224, 224)
        dummy_output = self.backbone(dummy_input)
        self.linear = nn.Linear(dummy_output.shape[-1], 1)
        # self.net.head.bias.data.fill_(0.3)  # approximate mean of the label

    def forward(self, x):
        out = self.backbone(x)
        out = self.linear(out).squeeze()
        return torch.sigmoid(out)
        # out = self.net(x).squeeze()
        # out = torch.sigmoid(out)
        # return out


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    m = RegressorNet()
    print(m(x))
    print(m(x).shape)
