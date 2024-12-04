import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


# 竞争神经网络
class CompetitiveNN:
    # 初始化
    def __init__(self, input_size, output_size):
        self.weights = np.full((input_size, output_size), 0.5)
        # self.weights = np.random.rand(input_size, output_size)
        self.labels = np.zeros(output_size)

    # 前向传播
    def forward(self, x):
        return np.dot(x, self.weights)

    # 训练
    def train(self, x, y, lr_initial=0.01, decay_rate=0.01, epochs=100):
        distances = np.zeros((epochs, self.weights.shape[1]))
        # 绘制初始权重
        self.plot_weights(-1, x, y)
        for epoch in range(epochs):
            lr = lr_initial * np.exp(-decay_rate * epoch)
            epoch_distances = np.zeros(self.weights.shape[1])
            win = np.zeros(self.weights.shape[1])
            for i in range(len(x)):
                outputs = self.forward(x[i])
                winner = np.argmax(outputs)
                self.weights[:, winner] += lr * (x[i] - self.weights[:, winner])
                epoch_distances[winner] = euclidean(x[i], self.weights[:, winner])
                win[winner] += 1
            distances[epoch] = epoch_distances
            print(f'训练次数：{epoch + 1}，学习速率：{lr:.8f}，平均欧几里得距离：{epoch_distances.mean():.8f}')
            print(f'胜利次数：{win}')
            if (epoch + 1) % 10 == 0:
                self.plot_weights(epoch, x, y)
        # 绘制迭代后权重
        self.plot_weights(epochs - 1, x, y)
        # 绘制欧几里得距离
        t = np.arange(epochs)
        for i in range(self.weights.shape[1]):
            plt.plot(t, distances[:, i])
        plt.plot(t, distances.mean(axis=1), color='r', linestyle='--')
        plt.xlabel('epoch')
        plt.ylabel('euclidean_distance')
        plt.show()

    # 绑定标签
    def determine_labels(self, x, y):
        result = np.zeros((self.weights.shape[1], self.weights.shape[1]))
        for i in range(len(x)):
            outputs = self.forward(x[i])
            winner = np.argmax(outputs)
            result[winner, y[i]] += 1
        for i in range(self.weights.shape[1]):
            print(f'神经元{i + 1}：{result[i]}')
            self.labels[i] = np.argmax(result[i])
        print(f'标签：{self.labels}')

    # 预测
    def predict(self, x):
        outputs = self.forward(x)
        return np.argmax(outputs, axis=1)

    # 绘制权值
    def plot_weights(self, epoch, f, l):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # 特征
        if l is not None:
            for i in range(len(f)):
                x = f[i][0]
                y = f[i][1]
                z = f[i][2]
                if l[i] == 0:
                    ax.scatter(x, y, z, color='r', alpha=0.2, marker='x')
                elif l[i] == 1:
                    ax.scatter(x, y, z, color='g', alpha=0.2, marker='x')
                else:
                    ax.scatter(x, y, z, color='b', alpha=0.2, marker='x')
        else:
            for i in range(len(f)):
                x = f[i][0]
                y = f[i][1]
                z = f[i][2]
                ax.scatter(x, y, z, color='k', alpha=0.2, marker='x')
            # 权值
        for i in range(self.weights.shape[1]):
            x = self.weights[0, i]
            y = self.weights[1, i]
            z = self.weights[2, i]
            ax.scatter(x, y, z, marker='o')
        ax.set_title(f'Epoch {epoch + 1}')
        plt.show()

class Dataset:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]