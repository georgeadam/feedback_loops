import numpy as np

from sklearn.model_selection import train_test_split

from .generic import TemporalDataWrapper, StaticDataWrapper


class LRETemporalDataWrapper(TemporalDataWrapper):
    def __init__(self, data, lre_val_proportion, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year):
        super().__init__(data, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year)

        self._lre_val_proportion = lre_val_proportion

    def get_val_data_lre(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        x_val_regular, x_val_lre, y_val_regular, y_val_lre = train_test_split(self._x_val, self._y_val,
                                                                              test_size=self._lre_val_proportion)

        np.random.set_state(random_state)

        return x_val_lre, y_val_lre

    def get_val_data_regular(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        x_val_regular, x_val_lre, y_val_regular, y_val_lre = train_test_split(self._x_val, self._y_val,
                                                                              test_size=self._lre_val_proportion)

        np.random.set_state(random_state)

        return x_val_regular, y_val_regular


class LREStaticDataWrapper(StaticDataWrapper):
    def __init__(self, data, lre_val_proportion, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, include_train, ddp, ddr, tvp, agg_data, num_updates)

        self._lre_val_proportion = lre_val_proportion

    def get_val_data_lre(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        x_val_regular, x_val_lre, y_val_regular, y_val_lre = train_test_split(self._x_val, self._y_val,
                                                                              test_size=self._lre_val_proportion)

        np.random.set_state(random_state)

        return x_val_lre, y_val_lre

    def get_val_data_regular(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        x_val_regular, x_val_lre, y_val_regular, y_val_lre = train_test_split(self._x_val, self._y_val,
                                                                              test_size=self._lre_val_proportion)

        np.random.set_state(random_state)

        return x_val_regular, y_val_regular
