# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


def load_and_preprocess_data():
    """
    加载Iris数据集并进行预处理
    返回：训练集、测试集、特征名称
    """
    # 加载Iris数据集
    iris = load_iris()
    X = iris.data  # 特征数据：4个特征，(150, 4)
    y = iris.target  # 标签数据：3个类别，(150,)

    # 划分训练集和测试集（70%训练，30%测试）
    # stratify=y保证训练集和测试集中各类别比例与原数据一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 特征标准化（均值为0，方差为1），提升模型训练稳定性
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # 训练集拟合并转换
    X_test = scaler.transform(X_test)  # 测试集仅转换

    # 转换为PyTorch张量（float32类型）
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)  # 分类任务标签用LongTensor
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, iris.feature_names


class SimpleMLP(nn.Module):
    """
    三层MLP神经网络：输入层(4) → 隐藏层(5) → 输出层(3)
    """

    def __init__(self):
        super(SimpleMLP, self).__init__()  # 调用父类构造函数

        # 定义网络层：使用Sequential容器按顺序堆叠层
        self.layers = nn.Sequential(
            nn.Linear(4, 5),  # 输入层→隐藏层：4个输入神经元，5个输出神经元
            nn.Sigmoid(),  # Sigmoid激活函数，引入非线性
            nn.Linear(5, 3)  # 隐藏层→输出层：5个输入神经元，3个输出神经元（对应3个类别）
        )

    def forward(self, x):
        """
        前向传播函数：定义数据在网络中的流动路径
        参数x：输入张量，形状为(batch_size, 4)
        返回：网络输出，形状为(batch_size, 3)
        """
        return self.layers(x)  # 数据通过所有层


def train_model(model, X_train, y_train, epochs=100, lr=0.1):
    """
    训练模型（带BP反向传播）
    参数：
        model: 要训练的模型
        X_train: 训练集特征
        y_train: 训练集标签
        epochs: 训练轮数
        lr: 学习率
    返回：训练好的模型、训练损失历史、训练准确率历史
    """
    # 定义损失函数：交叉熵损失（适用于多分类任务）
    criterion = nn.CrossEntropyLoss()
    # 定义优化器：随机梯度下降（SGD），用于更新参数
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 初始化记录训练过程的列表
    loss_history = []  # 记录每轮的损失值
    acc_history = []  # 记录每轮的准确率

    # 设置模型为训练模式（启用Dropout、BatchNorm等层的训练行为）
    model.train()

    # 开始训练循环
    for epoch in range(epochs):
        # 1. 前向传播：计算模型输出
        outputs = model(X_train)  # 输入数据通过模型得到输出

        # 2. 计算损失：比较模型输出与真实标签的差异
        loss = criterion(outputs, y_train)

        # 3. 反向传播：计算梯度
        optimizer.zero_grad()  # 清空之前的梯度（否则会累积）
        loss.backward()  # 反向传播计算参数梯度

        # 4. 更新参数：使用优化器根据梯度更新网络参数
        optimizer.step()

        # 计算训练准确率（用于监控训练过程）
        _, preds = torch.max(outputs, 1)  # 获取预测类别（取输出最大值对应的索引）
        accuracy = (preds == y_train).float().mean().item()  # 计算准确率

        # 记录本轮的损失和准确率
        loss_history.append(loss.item())
        acc_history.append(accuracy)

        # 每10轮打印一次训练状态
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    return model, loss_history, acc_history


def test_model(model, X_test, y_test):
    """
    测试模型性能
    参数：
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
    返回：测试准确率、预测结果
    """
    # 设置模型为评估模式（禁用Dropout、BatchNorm等层的训练行为）
    model.eval()

    # 使用torch.no_grad()禁用梯度计算（节省内存，加速计算）
    with torch.no_grad():
        outputs = model(X_test)  # 前向传播计算输出
        _, preds = torch.max(outputs, 1)  # 获取预测类别
        accuracy = (preds == y_test).float().mean().item()  # 计算准确率

    print(f'\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')
    return accuracy, preds


def visualize_training(loss_history, acc_history, X_test, y_test, preds, feature_names):
    """
    可视化训练过程和测试结果
    参数：
        loss_history: 训练损失历史
        acc_history: 训练准确率历史
        X_test: 测试集特征
        y_test: 测试集标签
        preds: 测试集预测结果
        feature_names: 特征名称
    """
    # 创建一个包含两个子图的画布
    plt.figure(figsize=(14, 6))

    # 第一个子图：训练损失和准确率变化曲线
    plt.subplot(1, 2, 1)
    # 绘制损失曲线（左y轴）
    plt.plot(loss_history, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss', color='b')
    plt.tick_params(axis='y', labelcolor='b')

    # 创建第二个y轴，绘制准确率曲线
    ax2 = plt.twinx()
    ax2.plot(acc_history, 'r-', label='Training Accuracy')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Training Process (BP)')
    plt.grid(alpha=0.3)

    # 第二个子图：测试集分类结果可视化（使用前两个特征）
    plt.subplot(1, 2, 2)
    # 定义颜色映射（三类样本用不同颜色）
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

    # 标记正确和错误的预测
    correct = preds == y_test
    # 绘制正确预测的样本（圆形标记）
    plt.scatter(X_test[correct, 0], X_test[correct, 1],
                c=y_test[correct], cmap=cmap, edgecolors='k', marker='o', s=50, label='Correct')
    # 绘制错误预测的样本（方形标记）
    plt.scatter(X_test[~correct, 0], X_test[~correct, 1],
                c=y_test[~correct], cmap=cmap, edgecolors='k', marker='s', s=50, label='Incorrect')

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Test Set Classification Results')
    plt.legend()

    # 调整子图间距
    plt.tight_layout()
    # 显示图形
    plt.show()


def main():
    """
    主函数：串联所有功能模块
    """
    # 1. 加载并预处理数据
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess_data()

    # 2. 创建模型实例
    print("\nInitializing MLP model...")
    model = SimpleMLP()

    # 3. 训练模型（带BP）
    print("\nTraining model with BP...")
    trained_model, loss_hist, acc_hist = train_model(model, X_train, y_train, epochs=100, lr=0.1)

    # 4. 测试模型
    print("\nTesting model...")
    test_acc, test_preds = test_model(trained_model, X_test, y_test)

    # 5. 可视化结果
    print("\nVisualizing results...")
    visualize_training(loss_hist, acc_hist, X_test, y_test, test_preds, feature_names)


# 程序入口
if __name__ == "__main__":
    main()
