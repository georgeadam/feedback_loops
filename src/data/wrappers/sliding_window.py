import copy
import math
import numpy as np

from sklearn.model_selection import train_test_split


class SlidingWindowStaticGenerator:
    def __init__(self, x, y, window_size, stride):
        self.x = x
        self.y = y

        self.window_size = window_size
        self.stride = stride
        self.num_updates = math.ceil((len(x) - window_size) / stride)
        print(self.num_updates)
        self.iteration = 0

    def __iter__(self):
        # start at 1 since the model was already trained on the very first window
        for i in range(1, self.num_updates + 1):
            idx_start = int((self.stride * i))
            # idx_end = int((self.stride * (i + 1)) + self.window_size)
            idx_end = idx_start + self.window_size

            self.iteration = i

            if idx_start > len(self.x):
                return

            yield self.x[idx_start: idx_end], self.y[idx_start: idx_end]

            if len(self.x) - idx_start < self.window_size:
                return


class SlidingWindowStaticWrapper:
    def __init__(self, data, batch_size, ddr, tvp, window_size, stride):
        self._batch_size = batch_size

        self._x = np.concatenate([data["x_train"], data["x_update"]])
        self._y_clean = np.concatenate([data["y_train"], data["y_update"]])
        self._y_corrupt = copy.deepcopy(self._y_clean)

        self._x_test = data["x_test"]
        self._y_test = data["y_test"]

        self._x_val = data["x_val"]
        self._y_val = data["y_val"]

        self._ddr = ddr
        self._tvp = tvp

        self._window_size = window_size
        self._stride = stride

        self._update_data = SlidingWindowStaticGenerator(self._x, self._y_clean, window_size, stride)
        self.dimension = self._x.shape[1]

    def get_ddr(self):
        return self._ddr

    def get_train_data(self):
        window_start = 0
        window_end = self._window_size

        x_train, y_train = self._x[window_start: window_end], self._y_clean[window_start: window_end]

        return x_train, y_train

    def get_init_train_data(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        window_start = 0
        window_end = self._window_size

        if self._tvp > 0:
            x_train, _, y_train, _ = train_test_split(self._x[window_start: window_end],
                                                      self._y_clean[window_start: window_end],
                                                      test_size=self._tvp)
        else:
            x_train, y_train = self._x[window_start: window_end], self._y_clean[window_start: window_end]

        np.random.set_state(random_state)

        return x_train, y_train

    def get_init_thresh_data(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        if self._tvp > 0:
            window_start = 0
            window_end = self._window_size

            _, x_thresh, _, y_thresh = train_test_split(self._x[window_start: window_end],
                                                        self._y_clean[window_start: window_end],
                                                        test_size=self._tvp)
        else:
            x_thresh, y_thresh = self._x_val, self._y_val

        np.random.set_state(random_state)

        return x_thresh, y_thresh

    def get_validation_data(self):
        return self._x_val, self._y_val

    def get_eval_data(self, *args):
        return self._x_test, self._y_test

    def get_update_data_generator(self):
        return self._update_data

    def store_current_update_batch_clean(self, x, y):
        pass

    def store_current_update_batch_corrupt(self, x, y):
        # window_start = self._update_data.iteration * self._stride
        window_end = (self._update_data.iteration * self._stride) + self._window_size
        window_start = window_end - self._stride

        self._y_corrupt[window_start: window_end] = y[-min(self._stride, len(self._y_corrupt) - window_start):]

    def get_all_data_for_model_fit_corrupt(self):
        window_start = self._update_data.iteration * self._stride
        window_end = (self._update_data.iteration * self._stride) + self._window_size

        return self._x[window_start: window_end], self._y_corrupt[window_start: window_end]

    def get_all_data_for_model_fit_clean(self):
        window_start = self._update_data.iteration * self._stride
        window_end = (self._update_data.iteration * self._stride) + self._window_size

        return self._x[window_start: window_end], self._y_clean[window_start: window_end]

    def get_all_data_for_threshold_fit(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        window_start = self._update_data.iteration * self._stride
        window_end = (self._update_data.iteration * self._stride) + self._window_size

        _, x_thresh, _, y_thresh = train_test_split(self._x[window_start: window_end],
                                                    self._y_corrupt[window_start: window_end],
                                                    test_size=self._tvp)

        np.random.set_state(random_state)

        return x_thresh, y_thresh

    def get_all_data_for_scaler_fit(self):
        random_state = np.random.get_state()
        np.random.seed(1)

        window_start = self._update_data.iteration * self._stride
        window_end = (self._update_data.iteration * self._stride) + self._window_size

        _, x_thresh = train_test_split(self._x[window_start: window_end],
                                       test_size=self._tvp)

        np.random.set_state(random_state)

        return x_thresh

    def accumulate_update_data(self):
        pass

    def get_cumulative_update_data(self):
        iteration = self._update_data.iteration

        idx_start = int((self._stride * iteration))
        idx_end = int((self._stride * iteration) + self._window_size)

        return self._x[idx_start: idx_end], self._y_clean[idx_start: idx_end]