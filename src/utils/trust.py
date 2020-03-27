def full_trust(*args, **kwargs):
    return 1.0


def conditional_trust(model_fpr=0.0, **kwargs):
    return 1 - model_fpr


def constant_trust(clinician_trust=1.0, **kwargs):
    return clinician_trust
