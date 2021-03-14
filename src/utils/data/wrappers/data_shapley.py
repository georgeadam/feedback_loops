from .generic import TemporalDataWrapper, StaticDataWrapper


class DataShapleyTemporalDataWrapper(TemporalDataWrapper):
    def __init__(self, data, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year):
        super().__init__(data, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year)


class DataShapleyStaticDataWrapper(StaticDataWrapper):
    def __init__(self, data, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, include_train, ddp, ddr, tvp, agg_data, num_updates)
