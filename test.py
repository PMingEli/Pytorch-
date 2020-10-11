import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import transforms
from torchsummary import summary

from hwdb import HWDB
from model import ConvNet


def valid(epoch, net, test_loarder, writer):
    print("开始验证...")
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loarder:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('correct number: ', correct)
        print('totol number:', total)
        acc = 100 * correct / total
        print('识别准确率为：%f%%' % (acc))


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

    # summary(net, input_size=(3, 64, 64), device='cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    writer = SummaryWriter(log_path)
    valid(0, net, testloader, writer=writer)
    
