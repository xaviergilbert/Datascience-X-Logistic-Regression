import sys
import os
import numpy as np
import copy
from data_treatment_class import Data_treatment

class Data_train:
    def __init__(self, X, Y, alpha=0.001, max_iter=7):
        self.alpha = alpha
        self.max_iter = max_iter
        self.X = X
        self.Y = Y
        self.Y_workable()
        self.Y_Gryffindor = self.is_house(1)
        self.Y_Hufflepuff = self.is_house(2)
        self.Y_Ravenclaw = self.is_house(3)
        self.Y_Slytherin = self.is_house(4)

    def Y_workable(self):
        houses = {'Gryffindor': 1, 'Hufflepuff': 2, 'Ravenclaw': 3, 'Slytherin': 4}
        for ligne in self.Y:
            ligne[0] = houses[ligne[0]]

    def is_house(self, house):
        y = copy.copy(self.Y)
        for ligne in y:
            if ligne[0] != house:
                ligne[0] = 0
            else:
                ligne[0] = 1
        return y

    def sigmoid_(self, x):
        x = np.array(x, dtype=np.float32)
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        """
            Predict class labels for samples in x_train.
            Arg:
                x_train: a 1d or 2d numpy ndarray for the samples
            Returns:
                y_pred, the predicted class label per sample.
                None on any error.
            Raises:
                This method should not raise any Exception.
        """
        tmp = np.dot(X, self.thetas[1:]) + self.thetas[:1]
        Y_pred = np.array(self.sigmoid_(tmp))
        return Y_pred

    def fit(self, X, Y):
        """
            Fit the model according to the given training data.
            Args:
                X: a 1d or 2d numpy ndarray for the samples
                Y: a scalar or a numpy ndarray for the correct labels
            Returns:
                self : object
                None on any error.
            Raises:
                This method should not raise any Exception.
        """
        self.thetas = np.zeros(X.shape[1] + 1)
        for i in range(self.max_iter):
            Y_pred = self.predict(X)
            Y_pred = np.reshape(Y_pred, np.shape(Y))
            self.thetas[0] -= (self.alpha * np.mean(Y_pred - Y, axis = 0))
            self.thetas[1:] = self.thetas[1:] - (self.alpha * np.mean((Y_pred - Y) * X, axis = 0))


def main():
    file = sys.argv[1]
    df = Data_treatment(file)
    X = df.normalize_data[:, 1:]
    Y = df.clean_data[:, :1]

    if os.path.exists("thetas.txt"):
        os.remove("thetas.txt")
    f = open("thetas.txt", "a+")
    model = Data_train(X, Y)

    model.fit(model.X, model.Y_Gryffindor)
    lines_of_text = str(model.thetas)[1:-1]
    f.write(lines_of_text)
    f.write('\\')
    model.fit(model.X, model.Y_Hufflepuff)
    lines_of_text = str(model.thetas)[1:-1]
    f.write(lines_of_text)
    f.write('\\')
    model.fit(model.X, model.Y_Ravenclaw)
    lines_of_text = str(model.thetas)[1:-1]
    f.write(lines_of_text)
    f.write('\\')
    model.fit(model.X, model.Y_Slytherin)
    lines_of_text = str(model.thetas)[1:-1]
    f.write(lines_of_text)

    f.close()


if __name__ == "__main__":
    main()
