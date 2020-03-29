from sklearn.preprocessing import StandardScaler


class NullScaler(object):
    def __init__(self):
        pass

    def fit(self, *args):
        pass

    def transform(self, x):
        return x


def get_scaler(normalization):
    if normalization:
        return StandardScaler()
    else:
        return NullScaler()