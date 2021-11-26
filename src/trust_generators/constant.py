class Constant(object):
    def __init__(self, expert_trust):
        self.expert_trust = expert_trust

    def __call__(self, *args, **kwargs) -> float:
        return self.expert_trust