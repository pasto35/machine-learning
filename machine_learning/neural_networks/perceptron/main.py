import torch
from torch import nn
import torch.optim as optim

from machine_learning.neural_networks.perceptron.perceptron import Perceptron

if __name__ == '__main__':
    input_size = 2
    model = Perceptron(input_size)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = torch.tensor([[0.0], [1.0], [1.0], [1.0]])

    epochs = 2000
    for epoch in range(epochs):
        model.train()

        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        test_output = model(data)
        predicted = test_output.round()
        print(f"Prediction outputs for OR gate:\n{predicted}")

