import copy
import numpy as np

from .metrics import eval_model


def get_update_fn(args):
    if args.update_type == "feedback":
        return update_model_feedback
    elif args.update_type == "no_feedback":
        return update_model_no_feedback
    elif args.update_type == "feedback_confidence":
        return update_model_feedback_confidence


def update_model_no_feedback(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                             intermediate=False, threshold=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)

    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        new_model.partial_fit(sub_x, sub_y, classes)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_feedback(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                          intermediate=False, threshold=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)

    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        sub_pred = new_model.predict(sub_x)
        fp_idx = np.logical_and(sub_y == 0, sub_pred == 1)
        sub_y[fp_idx] = 1

        new_model.partial_fit(sub_x, sub_y, classes)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_noise(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                       intermediate=False, threshold=None, rate=0.2):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)

    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        neg_idx = np.where(sub_y == 0)[0]

        if len(neg_idx) > 0:
            fp_idx = np.random.choice(neg_idx, int(rate * len(sub_y)))
            sub_y[fp_idx] = 1

        new_model.partial_fit(sub_x, sub_y, classes)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_conditional_trust(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                   intermediate=False, threshold=None, physician_fpr=0.1):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)

    trust = None

    trusts = []
    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1

        if trust is None:
            trust = 1 - float(len(model_fp_idx)) / float(len(sub_y))

        trusts.append(trust)

        physician_pred = copy.deepcopy(sub_y)
        neg_idx = np.where(physician_pred == 0)[0]
        physician_fp_idx = np.random.choice(neg_idx, int(physician_fpr * len(sub_y)))
        physician_pred[physician_fp_idx] = 1

        bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])

        target = bernoulli * model_pred + (1 - bernoulli) * physician_pred

        new_model.partial_fit(sub_x, target, classes)
        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]

        fpr = float(len(model_fp_idx)) / float(len(sub_y))

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

        trust = 1 - fpr

    return new_model, rates, trusts


def update_model_increasing_trust(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                  intermediate=False, threshold=None, trusts=[], physician_fpr=0.1):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)
    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        trust = trusts[i]

        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1

        physician_pred = copy.deepcopy(sub_y)
        neg_idx = np.where(physician_pred == 0)[0]

        if len(neg_idx) > 0:
            physician_fp_idx = np.random.choice(neg_idx, min(len(neg_idx), int(physician_fpr * len(sub_y))))
            physician_pred[physician_fp_idx] = 1

        bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])

        target = bernoulli * model_pred + (1 - bernoulli) * physician_pred

        new_model.partial_fit(sub_x, target, classes)
        model_pred = new_model.predict(sub_x)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_feedback_with_training(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                        intermediate=False, threshold=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)
    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        sub_pred = new_model.predict(sub_x)
        fp_idx = np.logical_and(sub_y == 0, sub_pred == 1)
        sub_y[fp_idx] = 1

        new_model.partial_fit(np.concatenate((sub_x, x_train)), np.concatenate((sub_y, y_train)), classes)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_feedback_with_training_cumulative(model, x_train, y_train, x_update, y_update, x_test, y_test,
                                                   num_updates, intermediate=False, threshold=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)

    cumulative_x = None
    cumulative_y = None

    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        sub_pred = new_model.predict(sub_x)
        fp_idx = np.logical_and(sub_y == 0, sub_pred == 1)
        sub_y[fp_idx] = 1

        if cumulative_x is None:
            cumulative_x = sub_x
            cumulative_y = sub_y
        else:
            cumulative_x = np.concatenate((cumulative_x, sub_x))
            cumulative_y = np.concatenate((cumulative_y, sub_y))

        new_model.partial_fit(np.concatenate((cumulative_x, x_train)), np.concatenate((cumulative_y, y_train)), classes)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_full_fit_feedback(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                   intermediate=False, threshold=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    cumulative_x = None
    cumulative_y = None

    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        sub_pred = new_model.predict(sub_x)
        fp_idx = np.logical_and(sub_y == 0, sub_pred == 1)
        sub_y[fp_idx] = 1

        if cumulative_x is None:
            cumulative_x = sub_x
            cumulative_y = sub_y
        else:
            cumulative_x = np.concatenate((cumulative_x, sub_x))
            cumulative_y = np.concatenate((cumulative_y, sub_y))

        new_model.fit(np.concatenate((cumulative_x, x_train)), np.concatenate((cumulative_y, y_train)))

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def update_model_full_fit_no_feedback(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                      intermediate=False, threshold=None):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    cumulative_x = None
    cumulative_y = None

    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        if cumulative_x is None:
            cumulative_x = sub_x
            cumulative_y = sub_y
        else:
            cumulative_x = np.concatenate((cumulative_x, sub_x))
            cumulative_y = np.concatenate((cumulative_y, sub_y))

        new_model.fit(np.concatenate((cumulative_x, x_train)), np.concatenate((cumulative_y, y_train)))

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, None


def update_model_feedback_confidence(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates,
                                     intermediate=False, threshold=None, drop_proportion=0.8):
    np.random.seed(1)
    new_model = copy.deepcopy(model)

    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)

    rates = {"fpr": [], "tpr": [], "fnr": [], "tnr": []}

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        sub_pred = new_model.predict(sub_x)
        sub_prob = new_model.predict_proba(sub_x)

        if len(sub_pred.shape) > 1 and sub_pred.shape[1] > 1:
            sub_prob = sub_prob[np.arange(len(sub_prob)), sub_pred]
        else:
            sub_prob = sub_prob[np.arange(len(sub_prob)), sub_pred.astype(int)]

        sorted_idx = np.argsort(sub_prob)

        sub_prob = sub_prob[sorted_idx]
        sub_pred = sub_pred[sorted_idx]
        sub_y = sub_y[sorted_idx]
        sub_x = sub_x[sorted_idx]

        drop_idx = np.where(sub_pred == 1)[0]
        drop_idx = drop_idx[: int(drop_proportion * len(drop_idx))]
        keep_idx = np.delete(np.arange(len(sub_y)), drop_idx)

        sub_pred = sub_pred[keep_idx]
        sub_y = sub_y[keep_idx]
        sub_x = sub_x[keep_idx]

        fp_idx = np.logical_and(sub_y == 0, sub_pred == 1)
        sub_y[fp_idx] = 1

        new_model.partial_fit(sub_x, sub_y, classes)

        append_rates(intermediate, new_model, rates, threshold, x_test, y_test)

    return new_model, rates


def append_rates(intermediate, new_model, rates, threshold, x_test, y_test):
    if intermediate:
        if threshold is None:
            new_pred = new_model.predict(x_test)
        else:
            pred_prob = new_model.predict_proba(x_test)
            new_pred = pred_prob[:, 1] >= threshold

        updated_tnr, updated_fpr, updated_fnr, updated_tpr = eval_model(y_test, new_pred)
        rates["fpr"].append(updated_fpr)
        rates["tpr"].append(updated_tpr)
        rates["fnr"].append(updated_fnr)
        rates["tnr"].append(updated_tnr)