import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_digits
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

if __name__ == '__main__':
    digits = load_digits()

    # show first 5 numbers with their labels
    # plt.figure(figsize=(20,4))
    # for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    #     plt.subplot(1, 5, index + 1)
    #     plt.imshow(np.reshape(image, (8, 8)), cmap='gray')
    #     plt.title(f'Label: {label}')
    # plt.show()

    # x_train - features for train
    # x_test - labels for train
    # y_train - features for test
    # y_test - labels for test
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)
    # print(x_train.shape)

    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)

    # predict one (first) number
    print(model.predict(x_test[0].reshape(1, -1)))

    # Accuracy
    score = model.score(x_test, y_test)
    print(score)

    predictions = model.predict(x_test)
    cm = confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print(cm)