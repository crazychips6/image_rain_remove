import torch
import torch.optim as optim
from prn import Net
import torch.nn as nn
import os
from train_data import MyTrainDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练图像的路径
input_path = ''  # 图片地址
label_path = ''  # label地址

net = Net().to(device)

########超参数########
learning_rate = 1e-3
batch_size = 50
epoch = 10  # 训练次数
####################

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_f = nn.MSELoss()

dataset_train = MyTrainDataset(input_path, label_path)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

if os.path.exists('./model.pth'):  # 判断模型有没有提前训练过
    print("继续训练!")
    net.load_state_dict(torch.load('./model.pth'))  # 加载训练过的模型
else:
    print("从头训练!")

for i in range(epoch):
    net.train()

    for j, (x, y) in enumerate(train_loader):  # 加载训练数据
        inputs = Variable(x).to(device)
        label = Variable(y).to(device)

        net.zero_grad()
        optimizer.zero_grad()
        output = net(inputs)
        loss = loss_f(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("已完成第{}次训练的{:.3f}%,目前损失值为{:.6f})".format(i + 1, ((j + 1) / 252) * 100, loss))
        if j % 9 == 0:
            torch.save(net.state_dict(), 'model.pth')  # 保存训练模型
