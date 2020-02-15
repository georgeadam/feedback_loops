from sklearn.metrics import confusion_matrix


def eval_model(y, y_pred):
    c_matrix = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = c_matrix.ravel()

    samples = float(len(y_pred))

    return tn / samples, fp / samples, fn / samples, tp / samples
