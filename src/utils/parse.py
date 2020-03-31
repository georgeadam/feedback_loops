import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def percentage(v):
    try:
        v = float(v)
    except:
        print("Could not convert argument: {} to float".format(v))

    if isinstance(v, float) and v >= 0.0 and v <= 1.0:
       return v
    else:
        raise argparse.ArgumentTypeError('Percentage value expected. But {} was given'.format(type(v)))


def str2none(v):
    if v is None or (isinstance(v, str) and v.lower() == "none"):
        return None
    elif isinstance(v, str):
        return v
    else:
        raise argparse.ArgumentTypeError("Expected str or None, but was given {}".format(type(v)))