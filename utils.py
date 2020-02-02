import copy
import numpy as np
from sklearn.metrics import confusion_matrix

from scipy.stats import multivariate_normal

def eval_model(y, y_pred):
    c_matrix = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = c_matrix.ravel()

    samples = float(len(y_pred))
    
    return tn / samples, fp / samples, fn / samples, tp / samples


def update_model_no_feedback(model, x_update, y_update, x_test, y_test, num_updates, intermediate=False):
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
        
        if intermediate:
            new_pred = new_model.predict(x_test)
            updated_tnr, updated_fpr, updated_fnr, updated_tpr = eval_model(y_test, new_pred)
            rates["fpr"].append(updated_fpr)
            rates["tpr"].append(updated_tpr)
            rates["fnr"].append(updated_fnr)
            rates["tnr"].append(updated_tnr)
        
    return new_model, rates


def update_model_feedback(model, x_update, y_update, x_test, y_test, num_updates, intermediate=False):
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
        
        if intermediate:
            new_pred = new_model.predict(x_test)
            updated_tnr, updated_fpr, updated_fnr, updated_tpr = eval_model(y_test, new_pred)
            rates["fpr"].append(updated_fpr)
            rates["tpr"].append(updated_tpr)
            rates["fnr"].append(updated_fnr)
            rates["tnr"].append(updated_tnr)
        
    return new_model, rates


def update_model_noise(model, x, y, num_updates, rate):
    np.random.seed(1)
    new_model = copy.deepcopy(model)
    
    size = float(len(y)) / float(num_updates)

    classes = np.unique(y)

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y[idx_start: idx_end])

        neg_idx = np.where(sub_y == 0)[0]
        
        if len(neg_idx) > 0:
            fp_idx = np.random.choice(neg_idx, int(rate * len(sub_y)))
            sub_y[fp_idx] = 1

        new_model.partial_fit(sub_x, sub_y, classes)
        
    return new_model


def update_model_constant_trust(model, x, y, num_updates, trust, physician_fpr):
    np.random.seed(1)
    new_model = copy.deepcopy(model)
    
    size = float(len(y)) / float(num_updates)

    classes = np.unique(y)

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y[idx_start: idx_end])

        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1
        
        physician_pred = copy.deepcopy(sub_y)        
        neg_idx = np.where(physician_pred == 0)[0]
        physician_fp_idx = np.random.choice(neg_idx, int(physician_fpr * len(sub_y)))
        physician_pred[physician_fp_idx] = 1
        
        bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])
        
        target = bernoulli * model_pred + (1 - bernoulli) * physician_pred

        new_model.partial_fit(sub_x, target, classes)
        
    return new_model


def update_model_conditional_trust(model, x, y, num_updates, initial_trust, physician_fpr):
    np.random.seed(1)
    new_model = copy.deepcopy(model)
    
    size = float(len(y)) / float(num_updates)

    classes = np.unique(y)
    trust = initial_trust

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y[idx_start: idx_end])

        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1
        
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
        
        trust = 1 - 2 * fpr
        
    return new_model


def update_model_monotonically_increasing_trust(model, x, y, num_updates, initial_trust, physician_fpr):
    np.random.seed(1)
    new_model = copy.deepcopy(model)
    
    size = float(len(y)) / float(num_updates)

    classes = np.unique(y)
    
    trusts = np.linspace(initial_trust, 1.0, num_updates)

    for i in range(num_updates):
        trust = trusts[i]
        
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y[idx_start: idx_end])

        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1
        
        physician_pred = copy.deepcopy(sub_y)        
        neg_idx = np.where(physician_pred == 0)[0]
        physician_fp_idx = np.random.choice(neg_idx, int(physician_fpr * len(sub_y)))
        physician_pred[physician_fp_idx] = 1
        
        bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])
        
        target = bernoulli * model_pred + (1 - bernoulli) * physician_pred

        new_model.partial_fit(sub_x, target, classes)
        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        
    return new_model


def update_model_increasing_trust(model, x, y, num_updates, trusts, physician_fpr):
    np.random.seed(1)
    new_model = copy.deepcopy(model)
    
    size = float(len(y)) / float(num_updates)

    classes = np.unique(y)

    for i in range(num_updates):
        trust = trusts[i]
        
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x[idx_start: idx_end, :]
        sub_y = copy.deepcopy(y[idx_start: idx_end])

        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        model_pred = copy.deepcopy(sub_y)
        model_pred[model_fp_idx] = 1
        
        physician_pred = copy.deepcopy(sub_y)        
        neg_idx = np.where(physician_pred == 0)[0]
        physician_fp_idx = np.random.choice(neg_idx, int(physician_fpr * len(sub_y)))
        physician_pred[physician_fp_idx] = 1
        
        bernoulli = np.random.choice([0, 1], len(sub_y), p=[1 - trust, trust])
        
        target = bernoulli * model_pred + (1 - bernoulli) * physician_pred

        new_model.partial_fit(sub_x, target, classes)
        model_pred = new_model.predict(sub_x)
        model_fp_idx = np.where(np.logical_and(sub_y == 0, model_pred == 1))[0]
        
    return new_model


def update_model_feedback_with_training(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates, intermediate=False):
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
        
        if intermediate:
            new_pred = new_model.predict(x_test)
            updated_tnr, updated_fpr, updated_fnr, updated_tpr = eval_model(y_test, new_pred)
            rates["fpr"].append(updated_fpr)
            rates["tpr"].append(updated_tpr)
            rates["fnr"].append(updated_fnr)
            rates["tnr"].append(updated_tnr)
        
    return new_model, rates


def update_model_feedback_with_training_cumulative(model, x_train, y_train, x_update, y_update, x_test, y_test, num_updates, intermediate=False):
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
        
        if intermediate:
            new_pred = new_model.predict(x_test)
            updated_tnr, updated_fpr, updated_fnr, updated_tpr = eval_model(y_test, new_pred)
            rates["fpr"].append(updated_fpr)
            rates["tpr"].append(updated_tpr)
            rates["fnr"].append(updated_fnr)
            rates["tnr"].append(updated_tnr)
        
    return new_model, rates


def update_model_full_fit(model, x_train, y_train, x_update, y_update, num_updates):
    np.random.seed(1)
    new_model = copy.deepcopy(model)
    
    size = float(len(y_update)) / float(num_updates)
    
    cumulative_x = None
    cumulative_y = None

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
        
    return new_model


def update_model_feedback_linear_trend(model, x_update, y_update, num_updates, offset):
    np.random.seed(1)
    new_model = copy.deepcopy(model)
    
    size = float(len(y_update)) / float(num_updates)

    classes = np.unique(y_update)
    offsets = np.linspace(0, offset, num_updates)

    for i in range(num_updates):
        idx_start = int(size * i)
        idx_end = int(size * (i + 1))
        sub_x = x_update[idx_start: idx_end, :] + offsets[i]
        sub_y = copy.deepcopy(y_update[idx_start: idx_end])

        sub_pred = new_model.predict(sub_x)
        fp_idx = np.logical_and(sub_y == 0, sub_pred == 1)
        sub_y[fp_idx] = 1

        new_model.partial_fit(sub_x, sub_y, classes)
        
    return new_model


def perturb_labels_fp(y, rate=0.05):
    y_copy = copy.deepcopy(y)
    n_pert = int(len(y_copy) * rate)
    
    neg_idx = y_copy == 0
    neg_idx = np.where(neg_idx)[0]
    
    pert_idx = np.random.choice(neg_idx, n_pert, replace=False)
    
    y_copy[pert_idx] = 1
    
    return y_copy


def make_gaussian_data(m0, m1, s0, s1, n, p0, p1, features=2):
    neg_samples = np.random.multivariate_normal(m0 * np.ones(features), s0 * np.eye(features), int(n * p0))
    pos_samples = np.random.multivariate_normal(m1 * np.ones(features), s1 * np.eye(features), int(n * p1))
    
    x = np.concatenate((neg_samples, pos_samples))
    y = np.concatenate((np.zeros(len(neg_samples)), np.ones(len(pos_samples))))
    
    idx = np.random.choice(len(x), len(x), replace=False)
    x = x[idx]
    y = y[idx]
    
    return x, y


def make_trend_gaussian_data(m0, m1, s0, s1, n, features=2, uniform_range=[-3, 3]):
    x = np.random.uniform(uniform_range[0], uniform_range[1], (n, features))
    
    pdf0 = multivariate_normal(mean=np.ones(features) * m0, cov=np.eye(features) * s0).pdf(x)
    pdf1 = multivariate_normal(mean=np.ones(features) * m1, cov=np.eye(features) * s1).pdf(x)
    
    p_y = pdf1 / (pdf0 + pdf1)
    
    y = []
    
    for i in range(len(x)):
        y.append(np.random.choice([0, 1], p=[1 - p_y[i], p_y[i]]))
    
    y = np.array(y)
    idx = np.random.choice(len(x), len(x), replace=False)
    x = x[idx]
    y = y[idx]
    
    return x, y

