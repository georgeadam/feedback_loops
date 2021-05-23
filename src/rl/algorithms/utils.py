from .td import nfqi, q_learning


def get_algorithm(name):
    if name == "nfqi":
        return nfqi
    elif name == "q_learning":
        return q_learning