class Conditional(object):
    def __init__(self):
        pass

    def __call__(self, model_fpr: float, *args, **kwargs) -> float:
        return 1 - model_fpr