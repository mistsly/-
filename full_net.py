# 神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class Net(nn.Module):
    # 构造函数
    def __init__(self,opt):
        super(Net, self).__init__()
        # 卷积层三个参数：in_channel, out_channels, 5*5 kernal
        # self.con1 = nn.Conv2d(1, 116, 5)
        # self.con2 = nn.Conv2d(116, 100, 5)
        # 全连接层两个参数：in_channels, out_channels
        if opt == 'L':
            self.fc1 = nn.Linear(14, 500)
        elif opt == 'Rc':
            self.fc1 = nn.Linear(6, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 1)

    # 前向传播
    def forward(self, input_x):
        # 全连接层
        x = F.relu(self.fc1(input_x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x)
        return x


def train(x_std, Y, opt):
    # X_train, X_test, y_train, y_test = train_test_split(x_std, Y, test_size=0.2, random_state=123)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Net(opt).to(device)
    # 损失函数
    criterion = nn.MSELoss()
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    epoch_num = 1
    k = 10
    k_num = 0
    for train_index, test_index in KFold(k, shuffle=True, random_state=42).split(x_std, Y):
        print(f"fold_{k_num}")
        k_num += 1
        X_train, X_test = x_std[train_index], x_std[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        loss_list = []
        for epoch in range(epoch_num):
            y_train_list = []
            correct = 0
            total = 0
            run_loss = 0.0
            i = 0
            for (X_input, Y_label) in zip(X_train, y_train):
                X_input, Y_label = torch.from_numpy(X_input).float(), torch.Tensor([Y_label])
                X_input, Y_label = X_input.to(device), Y_label.to(device)
                optimizer.zero_grad()
                outputs = net(X_input)
                lossValue = criterion(outputs, Y_label)
                lossValue.backward()
                optimizer.step()
                y_train_list.append(outputs.item())
                run_loss += lossValue.item()
                i += 1
                if i == 180 - 1:
                    print('epoch[%d, %5d] loss : %.3f' % (epoch + 1, i + 1, run_loss / 160))
                    loss_list.append(run_loss / 160)
                    run_loss = 0.0

        plt.plot(y_train_list, color='red')
        plt.plot(y_train, color='blue')
        plt.savefig(f'./{opt}_pic/y_train_Y_labels_{k_num}.png', dpi=300)
        # plt.show()
        plt.clf()

        plt.plot(loss_list, color='red')
        plt.savefig(f'./{opt}_pic/loss_value_{k_num}.png', dpi=300)
        # plt.show()
        plt.clf()
        torch.save(net.state_dict(), f'{opt}_model_params.pth')
        print(f"{opt}_finished training!")

        # 预测
        with torch.no_grad():
            y_pre_list = []
            for X_input, y in zip(X_test, y_test):
                # print(X_input)
                x_test = torch.from_numpy(X_input).float()
                x_test = x_test.to(device)
                output = net(x_test)
                y_pre_list.append(output.item())
                print(f'pre:{output.item()}_truth:{y}')
        plt.plot(y_pre_list, color='red')
        plt.plot(y_test, color='blue')
        plt.savefig(f'./{opt}_pic/y_pre_Y_test_{k_num}.png', dpi=300)
        # plt.show()
        plt.clf()


def prediction(X_test, opt):
    """
    :param X_test: 需要从前端获取
    :param opt: 在主函数那里有解释
    :return: 预测结果,需要传到前端去
    """
    # todo
    print('prediction')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Net(opt).to(device)
    net.load_state_dict(torch.load(f'{opt}_model_params.pth'))
    with torch.no_grad():
        X_test = torch.from_numpy(X_test).float()
        X_test = X_test.to(device)
        output = net(X_test)
        output = output.item()
        if output > 1:
            output = 0.999
        elif output < 0:
            output = 0.001
    return output
