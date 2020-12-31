from flax import optim


def make_optimizer(optimizer: str,
                       learning_rate: float,
                       weight_decay: float = 5.0e-4):
    if optimizer == 'SGD':
        return optim.Optimizer(lerning_rate=learning_rate)
    elif optimizer == 'Momentum':
        return optim.Momentum(learning_rate=learning_rate,
                                  weight_decay=weight_decay)
    elif optimizer == 'Adam':
        return optim.Adam(learning_rate=learning_rate,
                              weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer spec.')

