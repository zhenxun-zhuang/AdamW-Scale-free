"""
Load the desired optimizer.
"""

from adam import Adam


def load_optim(params, optim_method, eta0, weight_decay):
    """
    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        optim_method: which optimizer to use, currently only support
            'AdamL2', 'AdamW', and 'AdamProx'.
        eta0: initial step size.
        weight_decay: weight decay factor.

    Outputs:
        an optimizer
    """

    if optim_method in ['AdamL2', 'AdamW', 'AdamProx']:
        optimizer = Adam(params=params, lr=eta0, weight_decay=weight_decay,
                         weight_decay_option=optim_method)
    else:
        raise ValueError("Unsupported optimizer: {}".format(optim_method))

    return optimizer
