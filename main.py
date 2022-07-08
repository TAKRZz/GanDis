import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 128

# 加（下）载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


class Generator(nn.Module):
    # 生成器由四个全连接层组成
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # 定义前向传播
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    # 判别器由四个全连接层组成
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # 定义前向传播
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


z_dim = 100

# train_data.size(): (-1, 28, 28)
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)

# 定义交叉熵损失函数
criterion = nn.BCELoss()

# 定义优化器
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)


def D_train(x):
    # ================================================================== #
    #                      训练判别模型                      #
    # ================================================================== #
    D.zero_grad()

    # 真实数据，标签为1
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
    # 计算real_损失
    # 使用公式 BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))，来计算realimage的判别损失
    # 其中第二项永远为零，因为real_labels == 1
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # 在对抗样本，标签为0
    z = Variable(torch.randn(bs, z_dim).to(device))
    # 生成模型根据随机输入生成fake_images（对抗样本）
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))
    # 使用公式 BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))，来计算fakeImage的判别损失
    # 其中第二项永远为零，因为fake_labels == 0
    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # 反向传播和优化
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(x):
    # ================================================================== #
    #                       训练生成模型                       #
    # ================================================================== #
    G.zero_grad()

    # 生成模型根据随机输入生成fake_images（标签为1）,然后判别模型进行判别
    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # 反向传播和优化
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


start = time.time()
n_epoch = 10
loss_file = open("loss.txt", 'a')
for epoch in range(1, n_epoch + 1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        # 丢弃不满整个batch_size的数据
        if (len(x) != bs):
            continue
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))
    loss_file.write(
        '[{}/{}]: loss_d: {:.3f}, loss_g: {:.3f}\n'.format((epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)),
                                                           torch.mean(torch.FloatTensor(G_losses))))

end = time.time()
loss_file.write("Takes : {}s\n".format(end - start))
loss_file.close()

if __name__ == '__main__':
    print("JKL::")
