import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt

from hwdb1 import HWDB
from model import ConvNet


def valid(epoch, net, test_loarder, writer):
    # print("开始验证...")
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loarder:
            # print(labels)
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            # 取得分最高的那个类
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(predicted)
            value = predicted[0]
            if(value==3532):
                print("识别结果为：阿")
            # print(labels)


if __name__ == "__main__":
    # 超参数
    epochs = 20
    batch_size = 100
    lr = 0.01

    data_path = r'data'
    log_path = r'logs/batch_{}_lr_{}'.format(batch_size, lr)
    save_path = r'checkpoints/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 读取分类类别
    with open('char_dict', 'rb') as f:
        class_dict = pickle.load(f)
    num_classes = len(class_dict)

    # 读取数据
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = HWDB(path=data_path, transform=transform)
    print("测试集数据:", dataset.test_size)
    trainloader, testloader = dataset.get_loader(batch_size)

    net = ConvNet(num_classes)
    if torch.cuda.is_available():
        net = net.cuda()
    net.load_state_dict(torch.load('checkpoints/handwriting_iter_008.pth'))

    #summary(net, input_size=(3, 64, 64), device='cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    writer = SummaryWriter(log_path)
    valid(0, net, testloader, writer=writer)
    
