class LinearDecay:
    def __init__(self, epochs):
        self.epochs = epochs

    def step(self, epoch):
        return min(1 - ((1 / float(self.epochs)) * epoch), (1 / float(self.epochs)))


class ExponentialDecay:
    def __init__(self, epochs):
        self.scale = epochs / 3

    def step(self, epoch):
        return 10 ** (-epoch / (float(self.scale)))


def get_epsilon_decay_fn(name, epochs):
    if name == "linear":
        return LinearDecay(epochs)
    elif name == "exponential":
        return ExponentialDecay(epochs)