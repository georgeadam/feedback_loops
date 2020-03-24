def full_trust(*args, **kwargs):
    return 1.0


def conditional_trust(fpr):
    return 1 - fpr