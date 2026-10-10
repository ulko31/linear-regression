import numpy as np
from sklearn.model_selection import train_test_split


class GDRegressor:
    """
    Полиномиальная регрессия, обучаемая методом градиентного спуска.

    Модель выполняет стандартизацию признаков,
    создаёт полиномиальные признаки и обучает веса с помощью NumPy.
    """

    def __init__(
        self,
        alpha: float = 0.001,
        n_iter: int = 10000,
        degree: int = 5,
        progress: bool = False
    ):
        self.alpha = alpha
        self.n_iter = n_iter
        self.degree = degree
        self.progress = progress

        self.weights = None
        self.mean_ = None
        self.std_ = None
        self.poly_mean_ = None
        self.poly_std_ = None

    def fit(self, X_train, y_train):
        X = self._prepare_X(X_train, fit=True)
        y = self._prepare_y(y_train)

        self.weights = np.zeros((X.shape[1], 1))

        for i in range(self.n_iter):
            y_pred = X @ self.weights

            gradient = (
                2 / X.shape[0]
            ) * (X.T @ (y_pred - y))

            self.weights -= self.alpha * gradient

            if self.progress and (i + 1) % 1000 == 0:
                error = rmse(y, y_pred)
                print(
                    f"Iteration {i + 1}: "
                    f"RMSE = {error:.4f}"
                )

        return self

    def predict(self, X_test):
        if self.weights is None:
            raise ValueError("Model is not fitted yet")

        X = self._prepare_X(X_test, fit=False)

        return X @ self.weights

    def _prepare_X(self, X, fit=False):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if fit:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1

        X_scaled = (
            X - self.mean_
        ) / self.std_

        polynomial_features = [
            X_scaled ** power
            for power in range(1, self.degree + 1)
        ]

        X_poly = np.hstack(polynomial_features)

        if fit:
            self.poly_mean_ = X_poly.mean(axis=0)
            self.poly_std_ = X_poly.std(axis=0)
            self.poly_std_[self.poly_std_ == 0] = 1

        X_poly = (
            X_poly - self.poly_mean_
        ) / self.poly_std_

        bias = np.ones((X_poly.shape[0], 1))

        return np.hstack([bias, X_poly])

    @staticmethod
    def _prepare_y(y):
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return y


def z_scaler(feature):
    feature = np.asarray(feature, dtype=float)

    std = feature.std(axis=0)
    std = np.where(std == 0, 1, std)

    return (
        feature - feature.mean(axis=0)
    ) / std


def min_max(feature):
    feature = np.asarray(feature, dtype=float)

    min_value = feature.min(axis=0)
    max_value = feature.max(axis=0)

    denominator = np.where(
        max_value - min_value == 0,
        1,
        max_value - min_value
    )

    return (
        feature - min_value
    ) / denominator


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )


def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - y_true.mean()) ** 2
    )

    return 1 - ss_res / ss_tot


def find_optimal_params(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=18
    )

    candidates = [
        (10000, 0.0001),
        (20000, 0.0001),
        (30000, 0.0001),
        (5000, 0.0003),
        (10000, 0.0003),
        (20000, 0.0003),
        (5000, 0.001),
        (10000, 0.001),
    ]

    best_params = None
    best_score = -np.inf

    for n_iter, alpha in candidates:
        model = GDRegressor(
            alpha=alpha,
            n_iter=n_iter,
            degree=5
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        score = r_squared(
            y_val,
            y_pred
        )

        if score > best_score:
            best_score = score
            best_params = (n_iter, alpha)

    return best_params
