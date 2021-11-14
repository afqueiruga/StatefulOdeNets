import json
import os

import jax
from flax.linen import Module


def full_typename(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


def to_dict(obj):
    try:
        return {type(obj).__name__:{k:to_dict(v) for k,v in obj.__dict__.items()}}
    except AttributeError:
        return obj


def module_to_dict(module: Module, short=False):
    """Generate a dict representation of a module's arguments."""
    nester = (lambda x : x) if short else to_dict
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
        for k, v in module._state.children.items()  # pytype: disable=attribute-error
        if isinstance(v, Module)
    }

    if attributes:
        for attr in attributes.keys():
            value = getattr(module, attr)
            description[attr] = nester(value)

    if child_modules:
        for name, child in child_modules.items():
            child_description = module_dict(child, num_spaces)
            description[name] = child_description
    return {cls_name: description}


def module_to_single_line(module: Module):
    """Make a filename-friendly string of a module."""
    # TODO support nested modules
    dict_repr = module_to_dict(module, short=True)
    name = next(iter(dict_repr.keys()))  # There's only one at the top.
    attrs = ",".join(f"{k}={v}" for k, v in dict_repr[name].items())
    return f"{name}_{attrs}"[:500]


def parse_model_dict(dict_repr, scope):
    """Turn a dict into an instantiated model."""
    # assert len(dict_repr) == 1
    if type(dict_repr) is not dict:
        return dict_repr
    for name, fields in dict_repr.items():
        fields = {k: parse_model_dict(v, scope) for k, v in fields.items()}
        model = eval(name, scope)(**fields)  # TODO security
    return model


def load_model_dict_from_json(fname, scope):
    with open(fname, 'r') as f:
        dict_repr = json.loads(f.read())
    return parse_model_dict(dict_repr, scope)


def optimizer_def_to_dict(optimizer_def):
    return {full_typename(optimizer_def): optimizer_def.hyper_params.__dict__}


def parse_optimizer_def_dict(dict_repr, scope):
    return parse_model_dict(dict_repr, scope)


def load_optimizer_def_dict_from_json(fname, scope):
    return load_model_dict_from_json(fname, scope)


def count_parameters(tree):
    def size_of(x):
        try:
            return x.size
        except:
            try:
                return len(x)
            except:
                return 1
    return jax.tree_util.tree_reduce(lambda x, y : x + size_of(y), tree, initializer=0)
