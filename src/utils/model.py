from src.models.sklearn import lr, linear_svm
from src.models.pytorch import LR, LREWC, NN

def get_model_fn(model_name):
    if model_name == "lr":
        return lr
    elif model_name == "linear_svm":
        return linear_svm
    elif model_name == "lr_pytorch":
        return LR
    elif model_name == "lr_ewc":
        return LREWC
    elif model_name == "nn":
        return NN