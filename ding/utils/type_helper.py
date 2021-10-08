import typing
import treetensor
from collections import namedtuple
from typing import List, Dict, Tuple, TypeVar, Type

SequenceType = TypeVar('SequenceType', List, Tuple, namedtuple)
NestedType = TypeVar('NestedType', Dict, treetensor.torch.Tensor)
