# 우리가 신이고 우리가 인간 시뮬레이팅을 하는거야 그래서 인간한테 시련을 계속 내리는 게임을 하는거야 - 신류!!!

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


# 생성자는 랜덤 벡터 z를 입력받아서 가짜 이미지를 만들어낸다.

# 처음에는 이미지 자체가 랜덤하게 만들어졌기 때문에 알아보기 힘들 것이다.
# 하지만 지속적으로 구분자와 대결함에 따라 자신의 실력을 높혀나갈 것이고,
# 결국엔 특정 이미지(여기선, MNIST)와 비슷한 랜덤 벡터를 가진 이미지를 만들어 낼 것이다.
class Generator(nn.Module):

    # 네트워크 구조 생성
    def __init__(self):
        super(Generator, self).__init__()
        self.mainNet = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),  # 100차원의 랜덤 벡터를 받아서 256개로 늘린다.
            nn.LeakyReLU(0.2, inplace=True),  # 각 뉴런의 출력값이 00보다 높으면 그대로 전파하고, 낮으면 특정 숫자를 곱한다.
            nn.Linear(in_features=256, out_features=512),  # 256로 늘어난 이전 랜덤 벡터를 받아서 512개로 늘린다.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=28*28),
            nn.Tanh())  # 마지막 활성함수는 tanh을 이용한다.

    #  (batch_size * 100) 크기의 랜덤 벡터를 받아서
    #  이미지를 (batch_size * 1 * 28 * 28) 크기로 출력한다.
    def forward(self, *inputs):
        return self.mainNet(inputs).view(-1, 1, 28, 28)