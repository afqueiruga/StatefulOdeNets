from flax.linen import Module


def module_to_dict(module: 'Module'):
    """Returns a pretty printed representation of the module"""
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
