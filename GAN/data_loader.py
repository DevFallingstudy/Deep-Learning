import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

class LearningDataLoader:
    def __init__(self):
        # 데이터 전처리 방식 지정
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 데이터를 파이토치의 Tensor 형식으로 바꾼다.
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # 픽셀값 0~1 -> -1 ~ 1for
        ])

        # MNIST 데이터셋을 불러온다. 지정한 폴더에 없을 경우 자동으로 다운로드한다.
        self.mnist = datasets.MNIST(root='data', donwload=True, transform=transform)

        # 데이터를 한 번에 batch_size만큼 가져오는 dataloader를 생성한다.
        self.data_loader = DataLoader(mnist, batch_size=60, shuffle=True)

