import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt

class BinaryClassifier:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Binary Classifier")
        self.root.geometry("400x400")

        self.learning_rate_var = tk.DoubleVar(value=0.01)
        self.max_iterations_var = tk.IntVar(value=1000)

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="Learning Rate:").pack()
        ttk.Entry(self.root, textvariable=self.learning_rate_var).pack()

        ttk.Label(self.root, text="Max Iterations:").pack()
        ttk.Entry(self.root, textvariable=self.max_iterations_var).pack()

        ttk.Button(self.root, text="Train", command=self.train_classifier).pack()

        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()

    def train_classifier(self):
        learning_rate = self.learning_rate_var.get()
        max_iterations = self.max_iterations_var.get()

        # Dummy data points for demonstration
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        y = np.array([0, 0, 1, 1, 1])

        classifier = BinaryClassifier(learning_rate, max_iterations)
        classifier.train(X, y)

        self.plot_classification_region(classifier)

    def plot_classification_region(self, classifier):
        x_min, x_max = 0, 6
        y_min, y_max = 0, 6
        h = 0.01

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter([1, 2, 3], [1, 2, 3], color='blue')
        plt.scatter([4, 5], [4, 5], color='red')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
