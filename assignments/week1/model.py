import numpy as np


class LinearRegression:
    """
    A linear regression model that uses closed form to fit the model.
    """

    # w: np.ndarray
    # b: float

    def __init__(self):
        # raise NotImplementedError()
        self.b = 0
        self.w = np.array([0])
        self.lr = 0
        self.epochs = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # raise NotImplementedError()
        """
        Use closed-form to estimate.
        self.w=(X^TX)^{-1}X^Ty

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The label of the input data.

        Returns:
            None

        """
        X_new = (
            np.append(X.T, np.array([1] * X.shape[0])).reshape(
                X.shape[1] + 1, X.shape[0]
            )
        ).T

        w = np.dot(np.dot(np.linalg.inv(np.dot(X_new.T, X_new)), X_new.T), y)
        self.w = w[:-1:]
        self.b = w[-1]
        # print(self.b)
        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        # raise NotImplementedError()
        # print(X.shape)
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        y_hat = np.dot(X, self.w) + self.b
        # print(self.w)
        return y_hat


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.00000015, epochs: int = 600000
    ) -> None:
        """

        fit gradient descent
        y_i^{hat}=X_i*w+b
        Loss=1/N\\sum_i (y_i^{hat}-y)^2
        dw=2X(Xw+b-y)
        db=2sum(Xw+b-y)

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The label of the input data.

        Returns:
            None

        """
        # raise NotImplementedError()
        self.lr = lr
        self.epochs = epochs
        self.w = np.array([0] * X.shape[0])
        self.b = 0
        self.y = y
        # print(X.shape[1])
        for i in range(epochs):
            # print(self.w,self.b)
            Y_pred = self.predict(X)
            # print(Y_pred-y)
            dw = 2 * (np.matmul(X.T, Y_pred - y)) / X.shape[0]
            db = 2 * np.sum(Y_pred - y) / X.shape[0]
            # print(dw,db)
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db
        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        # raise NotImplementedError()
        # print(self.w,self.b)
        y_hat = np.dot(X, self.w) + self.b
        return y_hat
