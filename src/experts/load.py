from .oracle import Oracle
from .random_error import RandomError


def get_expert(args):
    if args.expert.type == "oracle":
        return Oracle()
    elif args.expert.type == "random_error":
        return RandomError(args.expert.fnr, args.expert.fpr)