class TraditionalMLTrainer:
    def __init__(self, model_fn, seed, warm_start, update, **kwargs):
        self._warm_start = warm_start
        self._update = update
        self._model_fn = model_fn

        self._seed = seed

    def initial_fit(self, model, data_wrapper, scaler):
        x_train, y_train = data_wrapper.get_init_train_data()

        model.fit(scaler.transform(x_train), y_train)

    def update_fit(self, model, data_wrapper, rate_tracker, scaler, *args, **kwargs):
        if not self._update:
            return model

        x, y = data_wrapper.get_all_data_for_model_fit_corrupt()

        if self._warm_start and hasattr(model, "partial_fit"):
            model.partial_fit(scaler.transform(x), y, classes=[0, 1])
        else:
            model = self._model_fn(data_wrapper.dimension)
            model.fit(scaler.transform(x), y)

        return model