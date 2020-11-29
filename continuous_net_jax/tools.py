from flax.linen import Module


def module_to_dict(module: Module):
    """Generate a dict representation of a module's arguments."""
    cls = type(module)
    cls_name = cls.__name__
    description = {}
    attributes = {
        k: v
        for k, v in cls.__annotations__.items()
        if k not in ('parent', 'name')
    }
    child_modules = {
        k: v
        for k, v in module.children.items()  # pytype: disable=attribute-error
        if isinstance(v, Module)
    }
    if attributes:
        for attr in attributes.keys():
            value = getattr(module, attr)
            description[attr] = value

    if child_modules:
        for name, child in child_modules.items():
            child_description = module_dict(child, num_spaces)
            description[name] = child_description
    return {cls_name: description}


def module_to_single_line(module: Module):
    """Make a filename-friendly string of a module."""
    # TODO support nested modules
    dict_repr = module_to_dict(module)
    name = next(iter(dict_repr.keys()))  # There's only one at the top.
    attrs = ",".join(f"{k}={v}" for k, v in dict_repr[name].items())
    return f"{name}_{attrs}"
