import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

if __name__ == '__main__':
    data = pd.read_csv("./../datasets/data_2.csv", sep=',', header=0)
    # print(len(data))
    # print(data.shape)
    # print(data.head())
    X = data.values[:,0:4]
    Y = data.values[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.3, random_state=100)
    dtc = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
    dtc.fit(x_train, y_train)
    prediction = dtc.predict(x_test)
    # Accuracy
    print(accuracy_score(y_test, prediction)*100)


    plt.figure(figsize=(15,8))
    plot_tree(dtc,
              feature_names=['Initial payment', 'Last payment', 'Credit Score', 'House Number'],
              class_names=['No', 'Yes'],
              filled=True)
    plt.show()