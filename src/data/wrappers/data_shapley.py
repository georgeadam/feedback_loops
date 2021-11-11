from .generic import TemporalDataWrapper, StaticDataWrapper


class DataShapleyTemporalDataWrapper(TemporalDataWrapper):
    def __init__(self, data, batch_size, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year)


class DataShapleyStaticDataWrapper(StaticDataWrapper):
    def __init__(self, data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates)


class MCShapleyDataWrapper(StaticDataWrapper):
    def __init__(self, data, batch_size, n_val, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates)

        self._n_val = n_val

    def get_val_data_shapley(self):
        return self._x_val[:self._n_val], self._y_val[:self._n_val]

    def get_val_data_regular(self):
        return self._x_val[:self._n_val], self._y_val[:self._n_val]


class MCShapleyTrainDataAsValidation(StaticDataWrapper):
    def __init__(self, data, batch_size, n_val, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates)

        self._n_val = n_val

    def get_val_data_shapley(self):
        return self._x_train, self._y_train

    def get_val_data_regular(self):
        return self._x_val[:self._n_val], self._y_val[:self._n_val]
