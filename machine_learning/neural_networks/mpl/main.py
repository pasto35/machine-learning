import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn

from machine_learning.neural_networks.mpl.model import Model

if __name__ == '__main__':
    torch.manual_seed(42)
    model = Model()

    df = pd.read_csv("./../../datasets/iris.csv")

    # Change last columns from Strings to Numbers
    df["variety"] = df["variety"].replace("Setosa", 0.0)
    df["variety"] = df["variety"].replace("Versicolor", 1.0)
    df["variety"] = df["variety"].replace("Virginica", 2.0)

    X = df.drop("variety", axis=1)
    y = df["variety"]

    # Convert to numpy array
    X = X.values
    y = y.values

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert X features to float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # Convert y labels to long tensors
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Set the criterion of model to measure the error, how far off the prediction s are from data
    criterion = nn.CrossEntropyLoss()
    #  Create Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train model
    epochs = 100
    losses = []
    for i in range(epochs):
        # go forward and get prediction
        y_pred = model.forward(X_train)

        # Measure the loss/error
        loss = criterion(y_pred, y_train)

        #  Keep track of our losses
        losses.append(loss.detach().numpy())

        if i % 10 == 0:
            print(f"Epoch: {i}, loss: {loss}")

        # Do back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
