from src.models.sklearn import lr, linear_svm


def get_model_fn(model_name):
    if model_name == "lr":
        return lr
    elif model_name == "linear_svm":
        return linear_svm