from flax import optim


def make_optimizer(optimizer: str, learning_rate: float = 0.001):
    if optimizer == 'SGD':
        return optim.Optimizer(lerning_rate=learning_rate)
    elif optimizer == 'Momentum':
        return optim.Momentum(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        return optim.Adam(learning_rate=learning_rate)
    else:
        raise ValueError('Unknown optimizer spec.')

