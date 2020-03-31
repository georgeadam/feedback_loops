import copy as copylib
from sklearn.preprocessing import StandardScaler


class NullScaler(object):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args):
        pass

    def transform(self, x):
        return x


class CustomStandardScaler(StandardScaler):
    def __init__(self, cols=None):
        super(CustomStandardScaler, self).__init__()
        self.cols = cols

    def fit(self, X, y=None):
        if self.cols is not None:
            super(CustomStandardScaler, self).fit(X[:, self.cols])
        else:
            super(CustomStandardScaler, self).fit(X)

    def transform(self, X, copy=None):
        if self.cols is not None:
            X_copy = copylib.deepcopy(X)
            X_copy[:, self.cols] = super(CustomStandardScaler, self).transform(X_copy[:, self.cols], copy=copy)

            return X_copy
        else:
            return super(CustomStandardScaler, self).transform(X)


def get_scaler(normalization, cols=None):
    if normalization:
        return CustomStandardScaler(cols)
    else:
        return NullScaler()
