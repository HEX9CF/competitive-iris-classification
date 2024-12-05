import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from network import CompetitiveNN
from dataset import Dataset

# 读取数据
def load_data():
    headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(filepath_or_buffer='../data/data.csv', header=None, names=headers)
    return data

# 归一化
def normalize(data):
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data

# 标准化
def standardize(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # data -= np.mean(data, axis=0)
    # data /= np.std(data, axis=0)
    return data

# 数据预处理
def preprocess(data, test_size):
    # 标签编码
    data['class'] = data['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # 分离特征和标签
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    # 标准化特征
    features = standardize(features)
    # for i in range(10):
    #     print(f'特征{i + 1}：{features[i]}，标签：{labels[i]}')

    # 划分训练集和测试集
    # train_features, test_features = features[:train_size], features[train_size:]
    # train_labels, test_labels = labels[:train_size], labels[train_size:]
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=1)

    print('训练集大小：', len(train_features))
    print('测试集大小：', len(test_features))

    train_data = Dataset(train_features, train_labels)
    test_data = Dataset(test_features, test_labels)

    return train_data, test_data

# 测试
def test(model, test_data):
    prediction = model.predict(test_data.features)

    print('测试结果：')
    for i in range(len(test_data)):
        print(f'测试数据{i + 1}，输入：{test_data.features[i]}，预测值：{model.labels[prediction[i]]}，真实值：{test_data.labels[i]}')

    accuracy = np.mean(model.labels[prediction] == test_data.labels) * 100
    print(f'准确率：{accuracy:.2f}%')

def main():
    data = load_data()
    # print(data)

    train_data, test_data = preprocess(data, 0.2)

    # 创建竞争神经网络
    model = CompetitiveNN(4, 3)

    # 训练竞争神经网络
    model.train(train_data.features, train_data.labels, lr_initial=0.5, decay_rate=0.0001, epochs=2000)

    model.mark_labels(test_data.features, test_data.labels)

    test(model, train_data)

    test(model, test_data)

if __name__ == '__main__':
    main()



