import numpy as np

from sklearn.model_selection import train_test_split

from .generic import TemporalDataWrapper, StaticDataWrapper, DataMiniBatcher


class LFETemporalDataWrapper(TemporalDataWrapper):
    def __init__(self, data, batch_size, lfe_val_proportion, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year)

        self._lfe_val_proportion = lfe_val_proportion

    def get_val_data_lre(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        x_val_regular, x_val_lre, y_val_regular, y_val_lre = train_test_split(self._x_val, self._y_val,
                                                                              test_size=self._lfe_val_proportion)

        np.random.set_state(random_state)

        return x_val_lre, y_val_lre

    def get_val_data_regular(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        x_val_regular, x_val_lre, y_val_regular, y_val_lre = train_test_split(self._x_val, self._y_val,
                                                                              test_size=self._lfe_val_proportion)

        np.random.set_state(random_state)

        return x_val_regular, y_val_regular


class LFEStaticDataWrapper(StaticDataWrapper):
    def __init__(self, data, batch_size, lfe_val_proportion, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates)

        self._lfe_val_proportion = lfe_val_proportion

    def get_val_data_lre(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        x_val_regular, x_val_lre, y_val_regular, y_val_lre = train_test_split(self._x_val, self._y_val,
                                                                              test_size=self._lfe_val_proportion)

        np.random.set_state(random_state)

        return x_val_lre, y_val_lre

    def get_val_data_regular(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        x_val_regular, x_val_lre, y_val_regular, y_val_lre = train_test_split(self._x_val, self._y_val,
                                                                              test_size=self._lfe_val_proportion)

        np.random.set_state(random_state)

        return x_val_regular, y_val_regular


class LFEValidationSizeTesting(StaticDataWrapper):
    def __init__(self, data, batch_size, n_val_reg, n_val_lre, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates)

        self._x_val_reg = self._x_val[: int(len(self._x_val) / 2)]
        self._y_val_reg = self._y_val[: int(len(self._y_val) / 2)]

        self._x_val_lre = self._x_val[int(len(self._x_val) / 2):]
        self._y_val_lre = self._y_val[int(len(self._y_val) / 2):]

        self._n_val_reg = n_val_reg
        self._n_val_lre = n_val_lre

    def get_val_data_lre(self):
        return self._x_val_lre[: self._n_val_lre], self._y_val_lre[: self._n_val_lre]

    def get_val_data_regular(self):
        return self._x_val_reg[: self._n_val_reg], self._y_val_reg[: self._n_val_reg]


class LFETrainDataAsValidationStatic(StaticDataWrapper):
    def __init__(self, data, batch_size, n_val, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates)

        self._n_val = n_val

    def get_val_data_lre(self):
        return self._x_train, self._y_train

    def get_val_data_regular(self):
        return self._x_val[:self._n_val], self._y_val[:self._n_val]


class LFETrainDataAsValidationTemporal(TemporalDataWrapper):
    def __init__(self, data, batch_size, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year)

    def get_val_data_lre(self):
        return self._x_train, self._y_train

    def get_val_data_regular(self):
        return self._x_val, self._y_val