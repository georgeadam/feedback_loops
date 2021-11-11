from .generic import StaticDataWrapper, TemporalDataWrapper


class RegularCSC2541BaselineDataWrapperStatic(StaticDataWrapper):
    def __init__(self, data, batch_size, n_val, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates)

        self._n_val = n_val

    def get_validation_data(self):
        return self._x_val[:self._n_val], self._y_val[:self._n_val]


class RegularCSC2541BaselineDataWrapperTemporal(TemporalDataWrapper):
    def __init__(self, data, batch_size, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year):
        super().__init__(data, batch_size, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year)