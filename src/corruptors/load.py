from .bias_oscillation import BiasOscillation
from .error_amplification import ErrorAmplification
from .identity import Identity
from .uniform_noise import UniformNoise

from omegaconf import DictConfig


def get_corruptor(args: DictConfig):
    if args.update_params.feedback == "amplify":
        return ErrorAmplification(args.update_params.fn_corruption_prob, args.update_params.fp_corruption_prob)
    elif args.update_params.feedback == "oscillate":
        return BiasOscillation(args.update_params.corruption_prob)
    elif args.update_params.feedback == "none":
        return Identity()
    elif args.update_params.feedback == "uniform_noise":
        return UniformNoise(args.update_params.noise)