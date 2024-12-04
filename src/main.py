import numpy as np
import pandas as pd

from model import CompetitiveNN, Dataset

# 读取数据
def load_data():
    headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(filepath_or_buffer='../data/data.csv', header=None, names=headers)
    # print(data)
    return data

# 数据预处理
def preprocess(data):
    # 标签编码
    data['class'] = data['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # 分离特征和标签
    features = data.iloc[:, :-1].values
    tags = data.iloc[:, -1].values

    # 归一化
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    # 划分训练集和测试集
    train_size = int(len(data) * 0.8)
    train_features, test_features = features[:train_size], features[train_size:]
    train_tags, test_tags = tags[:train_size], tags[train_size:]

    train_data = Dataset(train_features, train_tags)
    test_data = Dataset(test_features, test_tags)

    return train_data, test_data

# 测试
def test(model, test_data):
    prediction = model.predict(test_data.features)

    print('测试结果：')
    for i in range(len(test_data)):
        print(f'预测值：{prediction[i]}，真实值：{test_data.tags[i]}')

    accuracy = np.mean(prediction == test_data.tags) * 100
    print(f'准确率：{accuracy:.2f}%')


def main():
    data = load_data()
    train_data, test_data = preprocess(data)

    # 创建竞争神经网络
    model = CompetitiveNN(4, 3)

    # 训练竞争神经网络
    model.train(train_data.features, lr=0.01, epochs=10)

    test(model, test_data)

if __name__ == '__main__':
    main()



