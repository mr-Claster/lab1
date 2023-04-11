import numpy as np
import matplotlib.pyplot as plt


def calculate_loss(error):
   return np.mean(error ** 2)


def calculate_error(y, y_pred):
    return y - y_pred


def calculate_y_pred(weights, x):
    return weights[0] + weights[1] * x


def gradient_descent(x, y, learning_rate, weights):
    y_pred = calculate_y_pred(weights, x)
    error = calculate_error(y, y_pred)
    grad0 = -2 * np.mean(error)
    grad1 = -2 * np.mean(error * x)
    weights[0] -= learning_rate * grad0
    weights[1] -= learning_rate * grad1
    return weights


def print_data(weights, x, y):
    print(f"w0={weights[0]:.3f};\n"
              f"w1={weights[1]:.3f};\n"
              f"loss={calculate_loss(calculate_error(y, calculate_y_pred(weights, x))) :.3f}\n"
              f"=======================")


data = np.loadtxt("lab_1_train.csv", delimiter=",", skiprows=1)
xTrain = data[:, 1]
yTrain = data[:, 2]
weights = np.array([0.0, 0.0])

for epoch in range(0, 100000):
    weights = gradient_descent(xTrain, yTrain, 0.001, weights)
    print(f"Epoch {epoch + 1}:\n")
    print_data(weights, xTrain, yTrain)

data = np.loadtxt("lab_1_test.csv", delimiter=",", skiprows=1)
xTest = data[:, 1]
yTest = data[:, 2]
print_data(weights, xTest, yTest)

plt.scatter(xTrain, yTrain, color="black")
plt.scatter(xTest, yTest, color="blue")
x = np.array([x/10 for x in range(0, 11)])
plt.plot(x, weights[0] + weights[1] * x, color="red")
plt.show()
