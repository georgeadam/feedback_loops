from .generic import TemporalDataWrapper, StaticDataWrapper


class AUMTemporalDataWrapper(TemporalDataWrapper):
    def __init__(self, data, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year):
        super().__init__(data, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year)


class AUMStaticDataWrapper(StaticDataWrapper):
    def __init__(self, data, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(data, include_train, ddp, ddr, tvp, agg_data, num_updates)
