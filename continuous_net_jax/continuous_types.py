from typing import Any, Callable, Dict, List, Iterable, NamedTuple, Tuple, Union

# JAX doesn't have a good type system yet, so this is for readability.
ArrayType = Any
JaxTreeType = Union[ArrayType, Iterable['JaxTreeType'], Dict[Union[str, int],
                                                             'JaxTreeType']]

# Just the type signature of a normal jax function, or a flax.nn.Module.call.
RateEquation = Callable[[JaxTreeType, ArrayType], ArrayType]
# An instance of a depth function.
ContinuousParameters = Callable[[float], JaxTreeType]
# A general depth basis function, that returns depth functions.
BasisFunction = Callable[[Iterable[JaxTreeType]], ContinuousParameters]
# Integration scheme function.
IntegrationScheme = Callable[[float, RateEquation, float],
                             JaxTreeType]

