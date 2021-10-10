from collections import namedtuple
from typing import List, Dict, Tuple, TypeVar

import treetensor

SequenceType = TypeVar('SequenceType', List, Tuple, namedtuple)
NestedType = TypeVar('NestedType', Dict, treetensor.torch.Tensor)
