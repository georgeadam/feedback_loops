import argparse


def percentage(v):
    if isinstance(v, float) and v >= 0.0 and v <= 1.0:
       return v
    else:
        raise argparse.ArgumentTypeError('Percentage value expected.')