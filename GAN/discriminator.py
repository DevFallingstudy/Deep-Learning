# 비록 신류는 악한 존재일 수 있다고 하더라도, 우리는 그저 신류가 선하다는 '믿음'을 가질 수 밖에 없다.

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

# 구분자는 이미지를 입력받아 진짜인지 가짜인지를 확인하고 출력한다.

# 진짜일 확률을 0과 1사이의 숫자 중 하나로 출력한다(실수 형태).
# 생성자와 마찬가지로 4개의 선형 레이어를 쌓았으며, 활성 함수로 LeakyReLU를 사용했다.
# dropout은 학습 시에 무작위로 절반의 뉴런을 사용하지 않도록 한다.
# dropout을 통해서 과적합(overfitting)을 방지할 수도 있으며, 구분자의 학습 능률을 강제하여 생성자보다 빨리 학습되는 것 또한 막을 수 있다.
class Discriminator(nn.Module):

    # 네트워크 구조 생성
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mainNet = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=1024),  # 선형 레이어
            nn.LeakyReLU(0.2, inplace=True),  # 활성 함수
            nn.Dropout(0.5, inplace=True),  # 과적합 방지
            nn.Linear(in_features=28 * 28, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(in_features=28 * 28, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(in_features=28 * 28, out_features=1024),
            nn.Sigmoid())  # 출력값을 0과 1사이로 지정함

    # (batch_size * 1 * 28 * 28)
    def forward(self, *inputs):
        inputs = inputs.view(-1, 28*28)
        return self.mainNet(inputs)