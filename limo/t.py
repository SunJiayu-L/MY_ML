import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

#这段代码定义了一个名为 Net 的类，它是 PyTorch 中神经网络模型的基类 torch.nn.Module 的子类。
# 在深度学习中，通常会创建自定义的神经网络模型，通过继承 torch.nn.Module 类并重写其中的方法来实现。
class Net(torch.nn.Module):
#定义一个Net类，包含四个全连接层。输入为28*28像素

    def __init__(self):
        super().__init__()#super().__init__() 调用了父类 torch.nn.Module 的构造函数，确保在初始化过程中正确地设置了模型的基本属性。
        self.fc1 = torch.nn.Linear(28 * 28, 64)  #self.fc1 = torch.nn.Linear(28 * 28, 64) 定义了一个全连接层 (Linear)
                                                            # 输入大小为 28x28（即 MNIST 图像的大小），输出大小为 64。
        self.fc2 = torch.nn.Linear(64, 64)# 定义了第二个全连接层，输入大小为上一层的输出大小，输出大小为 64。
        self.fc3 = torch.nn.Linear(64, 64)#定义了第三个全连接层，同样的输入输出大小都为 64。
        self.fc4 = torch.nn.Linear(64, 10)# 定义了最后一个全连接层，将上一层的输出大小转换为 10，这适用于多类分类问题，比如对于 10 个数字的分类任务。

#前向传播
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))#将输入 x 传递给第一个全连接层 fc1，然后通过激活函数 ReLU 进行非线性变换。
        x = torch.nn.functional.relu(self.fc2(x))#将经过第一个全连接层和 ReLU 激活函数的结果传递给第二个全连接层 fc2，再次通过 ReLU 激活函数。
        x = torch.nn.functional.relu(self.fc3(x))#类似地，将结果传递给第三个全连接层 fc3 并进行 ReLU 激活。
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)#最后，将经过第四个全连接层 fc4 的结果传递给 softmax 函数，用于多类分类问题，并对输出取对数。
        return x


def get_data_loader(is_train):#用于导入数据。
    to_tensor = transforms.Compose([transforms.ToTensor()])#定义数据转化类型
    data_set = MNIST("", is_train, transform=to_tensor, download=True)#下载MINST数据
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):#评估神经网络
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:#从测试集中按批次取出数据
            outputs = net.forward(x.view(-1, 28 * 28))#计算神经网络预测值
            for i, output in enumerate(outputs):#对批次中每个结果进行比较，累计正确预测的数量
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total#返回正确率


def main():
    train_data = get_data_loader(is_train=True)#导入数据集
    test_data = get_data_loader(is_train=False)#导入测试集
    net = Net()#初始化神经网络

    print("initial accuracy:", evaluate(test_data, net))#打印初始正率
    #训练神经网络
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#
    #这行代码创建了一个 Adam 优化器对象，用于优化神经网络模型中所有可学习参数的值，即通过调整这些参数来最小化损失函数。
    #net.parameters() 返回了神经网络模型 net 中所有可学习参数的迭代器。
    #lr=0.001 指定了学习率，即每次优化步骤中参数更新的大小。Adam 优化算法的学习率通常设置为一个小的正数，以确保收敛性和稳定性。
    for epoch in range(10):
        for (x, y) in train_data:
            net.zero_grad()#初始化
            output = net.forward(x.view(-1, 28 * 28))#正向传播
            loss = torch.nn.functional.nll_loss(output, y)#计算差值
            loss.backward()#反向误差传播
            optimizer.step()#优化网络参数
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    #随机抽取三张图像，显示网络预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
