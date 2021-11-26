from .conditional import Conditional
from .confidence import Confidence
from .constant import Constant
from .full import Full


def get_trust_generator(args):
    if args.trust.type == "conditional_trust":
        return Conditional()
    elif args.trust.type == "confidence_trust":
        return Confidence()
    elif args.trust.type == "constant_trust":
        return Constant(args.trust.expert_trust)
    elif args.trust.type == "full_trust":
        return Full()