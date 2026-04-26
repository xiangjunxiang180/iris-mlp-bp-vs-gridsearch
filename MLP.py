import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# 1. 数据加载与预处理
def load_data():
    """加载Iris数据集，返回标准化后的训练/测试集"""
    iris = load_iris()
    X, y = iris.data, iris.target  # 4维特征，3类标签

    # 划分训练集(70%)和测试集(30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 特征标准化（均值0，方差1）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, iris.feature_names


# 2. 手写Sigmoid激活函数
def sigmoid(x):
    """Sigmoid激活函数，将值映射到0-1之间"""
    return 1 / (1 + np.exp(-x))


# 3. 手写三层MLP前向传播
def mlp_forward(x, w1, b1, w2, b2):
    """
    三层MLP前向传播
    x: 输入特征 (n_samples, 4)
    w1: 输入层→隐藏层权重 (4, 5)
    b1: 隐藏层偏置 (5,)
    w2: 隐藏层→输出层权重 (5, 3)
    b2: 输出层偏置 (3,)
    返回: 输出层结果 (n_samples, 3)
    """
    # 输入层→隐藏层：z1 = x·w1 + b1，激活后a1 = sigmoid(z1)
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    # 隐藏层→输出层：z2 = a1·w2 + b2，输出z2（未激活，直接用于预测）
    z2 = np.dot(a1, w2) + b2
    return z2


# 4. 网格搜索权重参数（非BP训练）+ 记录训练过程
def grid_search_weights(train_X, train_y, weight_range=(-0.8, 0.8), trials=50):
    """
    网格搜索最优权重（随机采样模拟网格）
    weight_range: 权重取值范围
    trials: 尝试次数
    返回: 最优权重和训练历史
    """
    # 初始化最佳参数和准确率
    best_acc = 0.0
    best_w1, best_b1, best_w2, best_b2 = None, None, None, None
    train_history = []  # 记录每次尝试的最佳准确率

    for i in range(trials):
        # 随机生成权重（模拟网格搜索的离散取值）
        # 输入层→隐藏层：4×5权重 + 5个偏置
        w1 = np.random.uniform(weight_range[0], weight_range[1], size=(4, 5))
        b1 = np.random.uniform(weight_range[0], weight_range[1], size=5)
        # 隐藏层→输出层：5×3权重 + 3个偏置
        w2 = np.random.uniform(weight_range[0], weight_range[1], size=(5, 3))
        b2 = np.random.uniform(weight_range[0], weight_range[1], size=3)

        # 前向传播计算输出
        outputs = mlp_forward(train_X, w1, b1, w2, b2)
        # 预测类别（取输出最大值索引）
        preds = np.argmax(outputs, axis=1)
        # 计算准确率
        acc = np.mean(preds == train_y)

        # 更新最佳参数
        if acc > best_acc:
            best_acc = acc
            best_w1, best_b1, best_w2, best_b2 = w1.copy(), b1.copy(), w2.copy(), b2.copy()

        # 记录当前最佳准确率
        train_history.append(best_acc)
        # 每10次尝试打印进度
        if (i + 1) % 10 == 0:
            print(f"尝试 {i + 1}/{trials}, 最佳训练准确率: {best_acc:.4f}")

    print(f"\n网格搜索最佳训练准确率: {best_acc:.4f}")
    return (best_w1, best_b1, best_w2, best_b2), train_history


# 5. 测试模型
def test_model(test_X, test_y, weights):
    """测试模型并返回准确率和预测结果"""
    w1, b1, w2, b2 = weights
    outputs = mlp_forward(test_X, w1, b1, w2, b2)
    preds = np.argmax(outputs, axis=1)
    acc = np.mean(preds == test_y)
    print(f"测试准确率: {acc * 100:.2f}%")
    return acc, preds


# 6. 可视化训练过程和分类结果
def visualize_results(train_history, test_X, test_y, preds, feature_names):
    """可视化训练过程和测试集分类结果"""
    plt.figure(figsize=(14, 6))

    # 子图1：训练过程中最佳准确率变化
    plt.subplot(121)
    plt.plot(range(1, len(train_history) + 1), train_history, 'b-', linewidth=2)
    plt.xlabel("尝试次数")
    plt.ylabel("最佳训练准确率")
    plt.title("权重网格搜索训练过程")
    plt.grid(alpha=0.3)
    plt.ylim(0.3, 1.0)  # 固定y轴范围

    # 子图2：测试集分类结果（前2维特征）
    plt.subplot(122)
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    # 正确样本（圆形）和错误样本（方形）
    correct = preds == test_y
    plt.scatter(test_X[correct, 0], test_X[correct, 1],
                c=preds[correct], cmap=cmap, edgecolors='k', marker='o', s=50, label='正确')
    plt.scatter(test_X[~correct, 0], test_X[~correct, 1],
                c=preds[~correct], cmap=cmap, edgecolors='k', marker='s', s=50, label='错误')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("测试集分类结果")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 主函数：串联所有功能
def main():
    # 1. 加载数据
    train_X, train_y, test_X, test_y, feature_names = load_data()

    # 2. 网格搜索权重参数（非BP）
    weights, train_history = grid_search_weights(train_X, train_y, trials=50)

    # 3. 测试模型
    acc, preds = test_model(test_X, test_y, weights)

    # 4. 可视化训练过程和结果
    visualize_results(train_history, test_X, test_y, preds, feature_names)


if __name__ == "__main__":
    main()
