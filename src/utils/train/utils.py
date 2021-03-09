from .nn import RegularNNTrainer, AUMNNTrainer
from .traditional_ml import TraditionalMLTrainer


from src.utils.model import TRADITIONAL_ML_MODEL_TYPES, NN_MODEL_TYPES

REGULAR_TRAIN_TYPES = ["regular", "aum", "data_shapley", "confidence", "sklearn", "nn_regular"]


def wrapped(constructor, **kwargs):
    def inside(*args, **specified_args):
        return constructor(*args, **kwargs, **specified_args)

    return inside


def get_trainer(args):
    if args.model.type in TRADITIONAL_ML_MODEL_TYPES:
        return wrapped(TraditionalMLTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update)
    elif args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_regular":
        return wrapped(RegularNNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optimizer=args.optim.optimizer, lr=args.optim.lr, momentum=args.optim.momentum,
                       nesterov=args.optim.nesterov, epochs=args.optim.epochs,
                       early_stopping_iter=args.optim.early_stopping_iter, weight_decay=args.optim.weight_decay,
                       device=args.optim.device)
    elif args.model.type in NN_MODEL_TYPES and args.optim_type == "nn_aum":
        return wrapped(AUMNNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optimizer=args.optim.optimizer, lr=args.optim.lr, momentum=args.optim.momentum,
                       nesterov=args.optim.nesterov, epochs=args.optim.epochs,
                       early_stopping_iter=args.optim.early_stopping_iter, weight_decay=args.optim.weight_decay,
                       device=args.optim.device)