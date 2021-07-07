from .nn import RegularNNTrainer, AUMNNTrainer, LRENNTrainer, LFENNTrainer, GradientShapleyNNTrainer, \
    PosPredNNTrainer, MonteCarloShapleyNNTrainer, DRONNTrainer, HausmanNNTrainer, PUNNTrainer, BalancedReweightNNTrainer
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
    elif args.model.type in NN_MODEL_TYPES and (args.optim.type == "nn_regular" or args.optim.type == "nn_regular_csc2541_baseline"):
        return wrapped(RegularNNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optim_args=args.optim)
    elif args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_aum":
        return wrapped(AUMNNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optim_args=args.optim)
    elif args.model.type in NN_MODEL_TYPES and (args.optim.type == "nn_lre" or
                                                args.optim.type == "nn_lre_val_set_size" or
                                                args.optim.type == "nn_lre_train_as_val"):
        return wrapped(LRENNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       regular_optim_args=args.optim.regular, lre_optim_args=args.optim.lre)
    elif args.model.type in NN_MODEL_TYPES and (args.optim.type == "nn_lfe"):
        return wrapped(LFENNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       regular_optim_args=args.optim.regular, lfe_optim_args=args.optim.lfe)
    elif args.model.type in NN_MODEL_TYPES and (args.optim.type == "nn_dro"):
        return wrapped(DRONNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optim_args=args.optim)
    elif args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_gradient_shapley":
        return wrapped(GradientShapleyNNTrainer, warm_start=args.update_params.warm_start,
                       update=args.update_params.do_update, regular_optim_args=args.optim.regular,
                       shapley_optim_args=args.optim.shapley)
    elif args.model.type in NN_MODEL_TYPES and (args.optim.type == "nn_mc_shapley" or args.optim.type == "nn_mc_shapley_train_as_val"):
        return wrapped(MonteCarloShapleyNNTrainer, warm_start=args.update_params.warm_start,
                       update=args.update_params.do_update, regular_optim_args=args.optim.regular,
                       shapley_optim_args=args.optim.shapley)
    elif args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_pos_pred":
        return wrapped(PosPredNNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optim_args=args.optim)
    elif args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_hausman":
        return wrapped(HausmanNNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optim_args=args.optim, rate_args=args.rates)
    elif args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_pu":
        return wrapped(PUNNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optim_args=args.optim)
    elif args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_balanced_reweight":
        return wrapped(BalancedReweightNNTrainer, warm_start=args.update_params.warm_start, update=args.update_params.do_update,
                       optim_args=args.optim)
