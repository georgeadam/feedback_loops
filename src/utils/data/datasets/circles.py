from sklearn.datasets import make_circles


def generate_circles_dataset(noise: float=0.0):
    def wrapped(n_train: int, n_update: int, n_test: int, num_features: int):
        x_train, y_train = make_circles(n_train, noise=noise)
        x_update, y_update = make_circles(n_update, noise=noise)
        x_test, y_test = make_circles(n_test, noise=0.0)

        return x_train, y_train, x_update, y_update, x_test, y_test, None

    return wrapped