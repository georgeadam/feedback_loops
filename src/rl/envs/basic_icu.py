import numpy as np

from gym import spaces
from gym.utils import seeding

ICU = 0
PALLIATIVE = 1

action_mapping = {0: "ICU", 1: "PALLIATIVE"}


class BasicICUEnvironment:
    def __init__(self, data, num_updates, scaler, icu_limit, include_icu, r_fn, r_fp, r_tn, r_tp, reward_fn, alpha, beta):
        self.x_train = data["x_train"]
        self.y_train = data["y_train"]

        self.x_val = data["x_val"]
        self.y_val = data["y_val"]

        self.x_test = data["x_test"]
        self.y_test = data["y_test"]

        self.x_update = data["x_update"]
        self.y_update = data["y_update"]

        self.x_update_corrupt = None
        self.y_update_corrupt = None

        self.x = None
        self.y = None

        self.scaler = scaler
        self._fit_scaler(self.scaler, self.x_train)

        self.icu_limit = icu_limit
        self.icu_capacity = 0
        self.release_delay = 0
        self.mode = "train"
        self.set_mode(self.mode)
        self.state = None
        self.idx = 0
        self.num_updates = num_updates
        self.include_icu = include_icu

        self.action_space = spaces.Discrete(2)
        self.r_fp = r_fp
        self.r_fn = r_fn
        self.r_tp = r_tp
        self.r_tn = r_tn

        self.reward_fn = reward_fn
        self.alpha = alpha
        self.beta = beta

        self.dimension = self._compute_dimension()

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        y = self.y[self.idx]

        # Stochastic outcomes for critical patients. Those sent to palliative can recover with probability alpha, those
        # sent to ICU can recover with probability beta
        if y == ICU and action == ICU:
            chance = np.random.choice([0, 1], 1, p=[self.beta, 1 - self.beta])
            y = chance
        elif y == ICU and action == PALLIATIVE:
            chance = np.random.choice([0, 1], 1, p=[self.alpha, 1 - self.alpha])
            y = chance

        done = self.idx == len(self.x) - 1
        reward = self._get_reward(action, y)

        # Increase ICU occupancy by 1 if action == ICU
        if action == ICU and self.icu_capacity < self.icu_limit:
            self.icu_capacity += 1

        if self.icu_capacity >= self.icu_limit:
            self.release_delay += 1

        if self.release_delay > 2:
            self.icu_capacity -= 1
            self.release_delay = 0
        # Release a patient from ICU with 50% probability
        elif self.icu_capacity > 0:
            release = np.random.choice([0, 1], p=[0.5, 0.5])

            if release:
                self.icu_capacity -= 1
                self.release_delay = 0

        if not done:
            self.idx += 1
            eta = min(self.icu_capacity / float(self.icu_limit), 1.0)
            eta = np.array(eta).reshape(1)

            self.state = self._build_state(eta, self.scaler.transform(self.x[self.idx]))

        label = self.y[self.idx]

        return self.state, reward, done, label

    def get_update_data(self, update_num):
        update_size = int(len(self.x_update) / float(self.num_updates))

        imputed_capacity = np.zeros(update_size).reshape(-1, 1)

        return np.concatenate([imputed_capacity, self.x_update[(update_num * update_size): ((update_num + 1) * update_size)]], axis=1), \
               self.y_update[(update_num * update_size): ((update_num + 1) * update_size)].copy()

    def store_update_data(self, x, y):
        if self.x_update_corrupt is None:
            self.x_update_corrupt = x
            self.y_update_corrupt = y
        else:
            self.x_update_corrupt = np.concatenate([self.x_update_corrupt, x])
            self.y_update_corrupt = np.concatenate([self.y_update_corrupt, y])

    def reset(self):
        # Randomly shuffle the training data
        shuffle_idx = np.arange(len(self.x)).astype(int)
        np.random.shuffle(shuffle_idx)

        self.x = self.x[shuffle_idx]
        self.y = self.y[shuffle_idx]
        self.idx = 0
        self.icu_capacity = 0

        eta = min(self.icu_capacity / float(self.icu_limit), 1.0)
        eta = np.array(eta).reshape(1)

        self.state = self._build_state(eta, self.scaler.transform(self.x[self.idx]))
        label = self.y[self.idx]

        return self.state, label

    def set_mode(self, mode):
        if mode == "train":
            self.mode = mode
            self.x = self.x_train
            self.y = self.y_train
        elif mode == "val":
            self.mode = mode
            self.x = self.x_val
            self.y = self.y_val
        elif mode == "test":
            self.mode = mode
            self.x = self.x_test
            self.y = self.y_test
        elif mode == "experiment":
            self.x = np.concatenate([self.x_train, self.x_update_corrupt])
            self.y = np.concatenate([self.y_train, self.y_update_corrupt])

    def _get_reward(self, action, y):
        if self.reward_fn == "negative":
            return self._negative_full_reward(action, y)
        elif self.reward_fn == "all":
            return self._all_full_reward(action, y)
        else:
            raise Exception("No such reward function exists: {}".format(self.reward_fn))

    def _negative_full_reward(self, action, y):
        eta = self.icu_capacity / float(self.icu_limit)

        r_full = ((self.r_fp ** 2) / (2 * np.sqrt((self.r_fp ** 2) + (3 * self.r_fp ** 2))) + self.r_fp)

        if self.icu_capacity == self.icu_limit:
            return r_full
        elif (y == ICU) and (action == PALLIATIVE):
            return - ((self.r_fp ** 2) / (2 * np.sqrt((self.r_fp ** 2) + ((3 * self.r_fp ** 2) * eta))) + self.r_fp)
        elif (y == PALLIATIVE) and (action == ICU):
            return - (((r_full - self.r_fn) * np.sqrt(eta)) + self.r_fn)
        else:
            return 0

    def _all_full_reward(self, action, y ):
        eta = self.icu_capacity / float(self.icu_limit)

        r_full = ((self.r_fp ** 2) / (2 * np.sqrt((self.r_fp ** 2) + (3 * self.r_fp ** 2))) + self.r_fp)

        if self.icu_capacity == self.icu_limit:
            return r_full
        elif (y == ICU) and (action == PALLIATIVE):
            return - ((self.r_fp ** 2) / (2 * np.sqrt((self.r_fp ** 2) + ((3 * self.r_fp ** 2) * eta))) + self.r_fp)
        elif (y == PALLIATIVE) and (action == ICU):
            return - (((r_full - self.r_fn) * np.sqrt(eta)) + self.r_fn)
        elif (y == ICU) and (y == action):
            return (self.r_tn / (np.sqrt(self.r_tn * eta + 0.1))) + self.r_tn
        elif (y == PALLIATIVE) and (y == action):
            return self.r_tp * np.sqrt(eta) + self.r_tp

    def _build_state(self, eta, x):
        if self.include_icu:
            return np.concatenate([eta, x])
        else:
            return x

    def _compute_dimension(self):
        if self.include_icu:
            return self.x_train.shape[1] + 1
        else:
            return self.x_train.shape[1]

    def _fit_scaler(self, scaler, x):
        scaler.fit(x)