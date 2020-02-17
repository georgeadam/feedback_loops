import argparse


def percentage(v):
    try:
        v = float(v)
    except:
        print("Could not convert argument: {} to float".format(v))

    if isinstance(v, float) and v >= 0.0 and v <= 1.0:
       return v
    else:
        raise argparse.ArgumentTypeError('Percentage value expected. But {} was given'.format(type(v)))