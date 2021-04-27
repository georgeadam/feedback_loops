import numpy as np

from sklearn.model_selection import train_test_split


class DataWrapper:
    def __init__(self, batch_size, include_train, ddp, ddr, tvp, agg_data):
        self._batch_size = batch_size
        self._include_train = include_train
        self._ddp = ddp
        self._ddr = ddr
        self._tvp = tvp
        self._agg_data = agg_data

        self._x_train = None
        self._y_train = None

        self._x_val = None
        self._y_val = None

        self._x_update_current_corrupt = None
        self._y_update_current_corrupt = None

        self._x_update_current_clean = None
        self._y_update_current_clean = None

        self._cumulative_x_update = None
        self._cumulative_y_update = None

        self._update_data = None

    def get_ddr(self):
        return self._ddr

    def get_train_data(self):
        return self._x_train, self._y_train

    def get_init_train_data(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        if self._tvp > 0:
            x_train, _, y_train, _ = train_test_split(self._x_train, self._y_train, test_size=self._tvp)
        else:
            x_train, y_train = self._x_train, self._y_train

        np.random.set_state(random_state)

        return x_train, y_train

    def get_init_thresh_data(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        if self._tvp > 0:
            _, x_thresh, _, y_thresh = train_test_split(self._x_train, self._y_train, test_size=self._tvp)
        else:
            x_thresh, y_thresh = self._x_train, self._y_train

        np.random.set_state(random_state)

        return x_thresh, y_thresh

    def get_validation_data(self):
        return self._x_val, self._y_val

    def get_update_data_generator(self):
        return self._update_data

    def get_cumulative_update_data(self):
        return self._cumulative_x_update, self._cumulative_y_update

    def accumulate_update_data(self):
        if self._agg_data:
            self._cumulative_x_update = np.concatenate((self._cumulative_x_update, self._x_update_current_corrupt))
            self._cumulative_y_update = np.concatenate((self._cumulative_y_update, self._y_update_current_corrupt))

    def store_current_update_batch_corrupt(self, x, y):
        self._x_update_current_corrupt = x
        self._y_update_current_corrupt = y

    def get_current_update_batch_corrupt(self):
        return self._x_update_current_corrupt, self._y_update_current_corrupt

    def store_current_update_batch_clean(self, x, y):
        self._x_update_current_clean = x
        self._y_update_current_clean = y

    def get_current_update_batch_clean(self):
        return self._x_update_current_clean, self._y_update_current_clean

    def _combine_all_data(self, corrupt=True):
        if corrupt:
            x_update_current = self._x_update_current_corrupt
            y_update_current = self._y_update_current_corrupt
        else:
            x_update_current = self._x_update_current_clean
            y_update_current = self._y_update_current_clean

        if self._include_train:
            all_train_x = np.concatenate([self._x_train, self._cumulative_x_update, x_update_current])
            all_train_y = np.concatenate([self._y_train, self._cumulative_y_update, y_update_current])
        else:
            all_train_x = np.concatenate([self._cumulative_x_update, x_update_current])
            all_train_y = np.concatenate([self._cumulative_y_update, y_update_current])

        data = {"all_train_x": all_train_x, "all_train_y": all_train_y, "all_thresh_x": None, "all_thresh_y": None}

        return data

    def _all_data_helper_train(self, corrupt=True):
        if corrupt:
            x_update_current = self._x_update_current_corrupt
            y_update_current = self._y_update_current_corrupt
        else:
            x_update_current = self._x_update_current_clean
            y_update_current = self._y_update_current_clean

        if self._include_train:
            x_threshold_set, x_threshold_reset, \
            y_threshold_set, y_threshold_reset = train_test_split(self._x_train, self._y_train, stratify=self._y_train,
                                                                  test_size=self._tvp)
            all_train_x = np.concatenate([x_threshold_set, self._cumulative_x_update, x_update_current])
            all_train_y = np.concatenate([y_threshold_set, self._cumulative_y_update, y_update_current])
            all_thresh_x = x_threshold_reset
            all_thresh_y = y_threshold_reset
        else:
            all_train_x = np.concatenate([self._cumulative_x_update, x_update_current])
            all_train_y = np.concatenate([self._cumulative_y_update, y_update_current])
            all_thresh_x = self._x_train
            all_thresh_y = self._y_train

        data = {"all_train_x": all_train_x, "all_train_y": all_train_y,
                "all_thresh_x": all_thresh_x, "all_thresh_y": all_thresh_y}

        return data

    def _all_data_helper_update_current(self, corrupt=True):
        if corrupt:
            x_update_current = self._x_update_current_corrupt
            y_update_current = self._y_update_current_corrupt
        else:
            x_update_current = self._x_update_current_clean
            y_update_current = self._y_update_current_clean

        neg_prop = np.sum(y_update_current == 0) / len(y_update_current)
        pos_prop = np.sum(y_update_current == 1) / len(y_update_current)

        if (int(neg_prop * self._tvp * len(y_update_current)) <= 1 or
                int(pos_prop * self._tvp * len(y_update_current)) <= 1):
            strat = None
        else:
            strat = self._y_update_current_corrupt

        x_threshold_set, x_threshold_reset, \
        y_threshold_set, y_threshold_reset = train_test_split(x_update_current, y_update_current,
                                                              stratify=strat, test_size=self._tvp)

        if self._include_train:
            all_train_x = np.concatenate([self._x_train, self._cumulative_x_update, x_threshold_set])
            all_train_y = np.concatenate([self._y_train, self._cumulative_y_update, y_threshold_set])
        else:
            all_train_x = np.concatenate([self._cumulative_x_update, x_threshold_set])
            all_train_y = np.concatenate([self._cumulative_y_update, y_threshold_set])

        all_thresh_x = x_threshold_reset
        all_thresh_y = y_threshold_reset

        data = {"all_train_x": all_train_x, "all_train_y": all_train_y, "all_thresh_x": all_thresh_x,
                "all_thresh_y": all_thresh_y}

        return data

    def _all_data_helper_update_cumulative(self, corrupt=True):
        if corrupt:
            x_update_current = self._x_update_current_corrupt
            y_update_current = self._y_update_current_corrupt
        else:
            x_update_current = self._x_update_current_clean
            y_update_current = self._y_update_current_clean

        temp_x = np.concatenate([self._cumulative_x_update, x_update_current])
        temp_y = np.concatenate([self._cumulative_y_update, y_update_current])

        x_threshold_set, x_threshold_reset, \
        y_threshold_set, y_threshold_reset = train_test_split(temp_x, temp_y, stratify=temp_y, test_size=self._tvp)

        if self._include_train:
            all_train_x = np.concatenate([self._x_train, x_threshold_set])
            all_train_y = np.concatenate([self._y_train, y_threshold_set])
        else:
            all_train_x = x_threshold_set
            all_train_y = y_threshold_set

        all_thresh_x = x_threshold_reset
        all_thresh_y = y_threshold_reset

        data = {"all_train_x": all_train_x, "all_train_y": all_train_y,
                "all_thresh_x": all_thresh_x, "all_thresh_y": all_thresh_y}

        return data

    def _all_data_helper_all(self, corrupt=True):
        if corrupt:
            x_update_current = self._x_update_current_corrupt
            y_update_current = self._y_update_current_corrupt
        else:
            x_update_current = self._x_update_current_clean
            y_update_current = self._y_update_current_clean

        if self._include_train:
            temp_x = np.concatenate([self._x_train, self._cumulative_x_update, x_update_current])
            temp_y = np.concatenate([self._y_train, self._cumulative_y_update, y_update_current])

            # x_threshold_set, x_threshold_reset, \
            # y_threshold_set, y_threshold_reset = train_test_split(temp_x, temp_y, stratify=temp_y, test_size=self._tvp)
            x_threshold_set, x_threshold_reset, \
            y_threshold_set, y_threshold_reset = train_test_split(temp_x, temp_y, test_size=self._tvp)
            all_train_x = x_threshold_set
            all_train_y = y_threshold_set
            all_thresh_x = x_threshold_reset
            all_thresh_y = y_threshold_reset
        else:
            neg_prop = (np.sum(y_update_current == 0) + np.sum(self._cumulative_y_update) == 0) / len(y_update_current)
            pos_prop = (np.sum(y_update_current == 1) + np.sum(self._cumulative_y_update == 1)) / len(y_update_current)

            if (int(neg_prop * self._tvp * len(y_update_current)) <= 1 or
                    int(pos_prop * self._tvp * len(y_update_current)) <= 1):
                strat = None
            else:
                strat = np.concatenate([self._cumulative_y_update, y_update_current])

            temp_x = np.concatenate([self._cumulative_x_update, x_update_current])
            temp_y = np.concatenate([self._cumulative_y_update, y_update_current])

            # x_threshold_set, x_threshold_reset, \
            # y_threshold_set, y_threshold_reset = train_test_split(temp_x, temp_y, stratify=strat, test_size=self._tvp)
            x_threshold_set, x_threshold_reset, \
            y_threshold_set, y_threshold_reset = train_test_split(temp_x, temp_y, test_size=self._tvp)

            all_train_x = x_threshold_set
            all_train_y = y_threshold_set

            if self._agg_data:
                all_thresh_x = np.concatenate([x_threshold_reset])
                all_thresh_y = np.concatenate([y_threshold_reset])
            else:
                all_thresh_x = np.concatenate([x_threshold_reset])
                all_thresh_y = np.concatenate([y_threshold_reset])

        data = {"all_train_x": all_train_x, "all_train_y": all_train_y,
                "all_thresh_x": all_thresh_x, "all_thresh_y": all_thresh_y}

        return data

    def get_all_data_for_model_fit_corrupt(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        if self._ddr is None:
            data = self._combine_all_data(True)
        elif self._ddp == "train":
            data = self._all_data_helper_train(True)
        elif self._ddp == "update_current":
            data = self._all_data_helper_update_current(True)
        elif self._ddp == "update_cumulative":
            data = self._all_data_helper_update_cumulative(True)
        elif self._ddp == "all":
            data = self._all_data_helper_all(True)

        np.random.set_state(random_state)

        return data["all_train_x"], data["all_train_y"]

    def get_all_data_for_model_fit_clean(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        if self._ddr is None:
            data = self._combine_all_data(False)
        elif self._ddp == "train":
            data = self._all_data_helper_train(False)
        elif self._ddp == "update_current":
            data = self._all_data_helper_update_current(False)
        elif self._ddp == "update_cumulative":
            data = self._all_data_helper_update_cumulative(False)
        elif self._ddp == "all":
            data = self._all_data_helper_all(False)

        np.random.set_state(random_state)

        return data["all_train_x"], data["all_train_y"]

    def get_all_data_for_threshold_fit(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        if self._ddr is None:
            data = self._combine_all_data()
        elif self._ddp == "train":
            data = self._all_data_helper_train()
        elif self._ddp == "update_current":
            data = self._all_data_helper_update_current()
        elif self._ddp == "update_cumulative":
            data = self._all_data_helper_update_cumulative()
        elif self._ddp == "all":
            data = self._all_data_helper_all()

        np.random.set_state(random_state)

        return data["all_thresh_x"], data["all_thresh_y"]

    def get_all_data_for_scaler_fit(self):
        if self._include_train:
            all_x = np.concatenate([self._x_train, self._cumulative_x_update, self._x_update_current_corrupt])
        else:
            all_x = np.concatenate([self._cumulative_x_update, self._x_update_current_corrupt])

        return all_x


class StaticUpdateDataGenerator():
    def __init__(self, x, y, num_updates):
        self.x = x
        self.y = y
        self.num_updates = num_updates

        if num_updates > 0:
            self.update_size = float(len(x)) / float(num_updates)
        else:
            self.update_size = 0

    def __iter__(self):
        for i in range(self.num_updates):
            idx_start = int(self.update_size * i)
            idx_end = int(self.update_size * (i + 1))

            yield self.x[idx_start: idx_end], self.y[idx_start: idx_end]


class StaticDataWrapper(DataWrapper):
    def __init__(self, data, batch_size, include_train, ddp, ddr, tvp, agg_data, num_updates):
        super().__init__(batch_size, include_train, ddp, ddr, tvp, agg_data)
        self._x_train = data["x_train"]
        self._y_train = data["y_train"]

        if "y_train_clean" in data.keys():
            self._y_train_clean = data["y_train_clean"]

        self._x_val = data["x_val"]
        self._y_val = data["y_val"]

        self._x_update = data["x_update"]
        self._y_update = data["y_update"]

        self._x_test = data["x_test"]
        self._y_test = data["y_test"]

        self._x_update_current = None
        self._y_update_current = None

        self._cumulative_x_update = np.array([]).astype(float).reshape(0, self._x_train.shape[1])
        self._cumulative_y_update = np.array([]).astype(int)

        self._update_data = StaticUpdateDataGenerator(self._x_update, self._y_update, num_updates)
        self.dimension = self._x_train.shape[1]

    def get_eval_data(self, *args):
        return self._x_test, self._y_test


class TemporalUpdateDataGenerator():
    def __init__(self, x, y, tyl, uyl, years):
        self._x = x
        self._y = y
        self._tyl = tyl
        self._uyl = uyl
        self._years = years

    def __iter__(self):
        for year in range(self._tyl + 1, self._uyl):
            update_idx = self._years == year

            yield self._x[update_idx], self._y[update_idx]


class TemporalDataWrapper(DataWrapper):
    def __init__(self, data, batch_size, include_train, ddp, ddr, tvp, agg_data, tyl, uyl, next_year):
        super().__init__(batch_size, include_train, ddp, ddr, tvp, agg_data)
        self._x_train = data["x_train"]
        self._y_train = data["y_train"]

        self._x_val = data["x_val"]
        self._y_val = data["y_val"]

        self._x_rest = data["x_rest"]
        self._y_rest = data["y_rest"]

        self._x_update_current = None
        self._y_update_current = None

        self._years_rest = data["years_rest"]
        self._current_year = tyl

        self._tyl = tyl
        self._uyl = uyl
        self._next_year = next_year

        self._cumulative_x_update = np.array([]).astype(float).reshape(0, self._x_train.shape[1])
        self._cumulative_y_update = np.array([]).astype(int)

        self._update_data = TemporalUpdateDataGenerator(self._x_rest, self._y_rest, tyl, uyl, self._years_rest)
        self.dimension = self._x_train.shape[1]

    def get_eval_data(self, update_num):
        if self._next_year:
            eval_idx = self._years_rest == self._tyl + update_num + 1
        else:
            eval_idx = self._years_rest == self._uyl

        x_eval, y_eval = self._x_rest[eval_idx], self._y_rest[eval_idx]

        return x_eval, y_eval


class DataMiniBatcher():
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

        if batch_size is not None:
            self.num_batches = max(int(float(len(x)) / float(batch_size)), 1)

    def __iter__(self):
        if self.batch_size is None:
            yield self.x, self.y
        else:
            random_indices = np.arange(len(self.x)).astype(int)
            np.random.shuffle(random_indices)

            for i in range(self.num_batches):
                idx_start = int(self.batch_size * i)
                idx_end = int(self.batch_size * (i + 1))

                indices = random_indices[idx_start: idx_end]

                yield self.x[indices], self.y[indices]