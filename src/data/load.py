from .datasets import generate_blobs_dataset
from .datasets import generate_circles_dataset
from .datasets import generate_gaussian_dataset
from .datasets import load_mimic_iii_data
from .datasets import load_mimic_iv_data, mimic_iv_paths
from .datasets import generate_moons_dataset
from .datasets import generate_sklearn_make_classification_dataset
from .datasets import generate_real_dataset_static
from .datasets import load_support2cls_data
from .datasets import generate_real_dataset_temporal
from .datasets import generate_adult_dataset
from .datasets import generate_credit_g_dataset
from .datasets import generate_compas_dataset

from .wrappers import StaticDataWrapper
from .wrappers import TemporalDataWrapper
from .wrappers import LRETemporalDataWrapper
from .wrappers import LREStaticDataWrapper
from .wrappers import LREValidationSizeTesting
from .wrappers import LRETrainDataAsValidationStatic
from .wrappers import LRETrainDataAsValidationTemporal
from .wrappers import LFEStaticDataWrapper
from .wrappers import LFETemporalDataWrapper
from .wrappers import AUMStaticDataWrapper
from .wrappers import AUMTemporalDataWrapper
from .wrappers import DataShapleyStaticDataWrapper
from .wrappers import DataShapleyTemporalDataWrapper
from .wrappers import RegularCSC2541BaselineDataWrapperStatic
from .wrappers import RegularCSC2541BaselineDataWrapperTemporal
from .wrappers import MCShapleyDataWrapper
from .wrappers import MCShapleyTrainDataAsValidation

from omegaconf import DictConfig
from src.models import TRADITIONAL_ML_MODEL_TYPES, NN_MODEL_TYPES
from src.utils.typing import DataFn


def get_data_fn(args: DictConfig) -> DataFn:
    if args.data.type == "gaussian":
        if hasattr(args.data, "m0"):
            return generate_gaussian_dataset(args.data.m0, args.data.m1,
                                             args.data.s0, args.data.s1,
                                             args.data.p0, args.data.p1)
        else:
            return generate_gaussian_dataset()
    elif args.data.type == "sklearn":
        return generate_sklearn_make_classification_dataset(args.data.noise)
    elif args.data.type == "mimic_iii":
        return generate_real_dataset_static(load_mimic_iii_data, balanced=args.data.balanced)
    elif "mimic_iv" in args.data.type:
        if args.data.temporal:
            return generate_real_dataset_temporal(load_mimic_iv_data,
                                                  mimic_iv_paths[args.data.type]["path"], args.data.balanced,
                                                  categorical=mimic_iv_paths[args.data.type]["categorical"], tyl=args.data.tyl,
                                                  model=args.model.type)
        else:
            return generate_real_dataset_static(load_mimic_iv_data,
                                                mimic_iv_paths[args.data.type]["path"], args.data.balanced,
                                                categorical=mimic_iv_paths[args.data.type]["categorical"],
                                                model=args.model.type)
    elif args.data.type == "moons":
        return generate_moons_dataset(args.data.start, args.data.end, args.data.noise)
    elif args.data.type == "circles":
        return generate_circles_dataset(args.data.noise)
    elif args.data.type == "support2":
        return generate_real_dataset_static(load_support2cls_data)
    elif args.data.type == "blobs":
        return generate_blobs_dataset(args.data.noise)
    elif args.data.type == "adult":
        return generate_adult_dataset(args.data.noise)
    elif args.data.type == "credit_g":
        return generate_credit_g_dataset(args.data.noise)
    elif args.data.type == "compas":
        return generate_compas_dataset(args.data.noise)


def wrap_constructor(constructor, **kwargs):
    def inner(data):
        return constructor(data, **kwargs)

    return inner


def get_data_wrapper_fn(args):
    if args.data.temporal:
        if args.model.type in TRADITIONAL_ML_MODEL_TYPES or (args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_regular"):
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, tyl=args.data.tyl,
                                    uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.model.type in TRADITIONAL_ML_MODEL_TYPES or (args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_regular_csc2541_baseline"):
            constructor = RegularCSC2541BaselineDataWrapperTemporal
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, tyl=args.data.tyl,
                                    uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_lfe":
            constructor = LFETemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, lfe_val_proportion=args.optim.lfe.val_proportion,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    tyl=args.data.tyl, uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_lre":
            constructor = LRETemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, lre_val_proportion=args.optim.lre.val_proportion,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    tyl=args.data.tyl, uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_lre_train_as_val":
            constructor = LRETrainDataAsValidationTemporal
            return wrap_constructor(constructor, batch_size=args.data.batch_size,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    tyl=args.data.tyl, uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_aum":
            constructor = AUMTemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    tyl=args.data.tyl, uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_gradient_shapley":
            constructor = DataShapleyTemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    tyl=args.data.tyl, uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_pos_pred":
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    tyl=args.data.tyl, uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_dro":
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, tyl=args.data.tyl,
                                    uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_hausman":
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, tyl=args.data.tyl,
                                    uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_pu":
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, tyl=args.data.tyl,
                                    uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_balanced_reweight":
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, tyl=args.data.tyl,
                                    uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_imbalanced":
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, tyl=args.data.tyl,
                                    uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_low_conf":
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, tyl=args.data.tyl,
                                    uyl=args.data.uyl, next_year=args.data.next_year)
        elif args.optim.type == "nn_soft_target":
            constructor = TemporalDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    tyl=args.data.tyl, uyl=args.data.uyl, next_year=args.data.next_year)
    else:
        if args.model.type in TRADITIONAL_ML_MODEL_TYPES or (args.model.type in NN_MODEL_TYPES and args.optim.type == "nn_regular"):
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_lfe":
            constructor = LFEStaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, lfe_val_proportion=args.optim.lfe.val_proportion,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_lre":
            constructor = LREStaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, lre_val_proportion=args.optim.lre.val_proportion,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_aum":
            constructor = AUMStaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_gradient_shapley":
            constructor = DataShapleyStaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_mc_shapley":
            constructor = MCShapleyDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, n_val=args.optim.regular.n_val,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_mc_shapley_train_as_val":
            constructor = MCShapleyTrainDataAsValidation
            return wrap_constructor(constructor, batch_size=args.data.batch_size, n_val=args.optim.regular.n_val,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_pos_pred":
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_lre_val_set_size":
            constructor = LREValidationSizeTesting
            return wrap_constructor(constructor, batch_size=args.data.batch_size, n_val_reg=args.optim.lre.n_val_reg,
                                    n_val_lre=args.optim.lre.n_val_lre,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_lre_train_as_val":
            constructor = LRETrainDataAsValidationStatic
            return wrap_constructor(constructor, batch_size=args.data.batch_size, n_val=args.optim.lre.n_val,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_regular_csc2541_baseline":
            constructor = RegularCSC2541BaselineDataWrapperStatic
            return wrap_constructor(constructor, batch_size=args.data.batch_size, n_val=args.optim.n_val,
                                    include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_dro":
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train,
                                    ddp=args.data.ddp, ddr=args.rates.ddr, tvp=args.data.tvp,
                                    agg_data=args.update_params.agg_data, num_updates=args.data.num_updates)
        elif args.optim.type == "nn_hausman":
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_pu":
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_balanced_reweight":
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_imbalanced":
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_low_conf":
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)
        elif args.optim.type == "nn_soft_target":
            constructor = StaticDataWrapper
            return wrap_constructor(constructor, batch_size=args.data.batch_size, include_train=args.update_params.include_train, ddp=args.data.ddp,
                                    ddr=args.rates.ddr, tvp=args.data.tvp, agg_data=args.update_params.agg_data,
                                    num_updates=args.data.num_updates)