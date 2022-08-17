from dataclasses_json import config
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import marshmallow_numpy
import numpy as np

__all__ = ['NumpyField']

@dataclass
class _NumpyArrayDTO(DataClassJsonMixin):
    dtype: str
    data: List[Any]

def ndarray_serialize(value: np.ndarray):
    if value is None:
        return None
    return asdict(_NumpyArrayDTO(dtype=value.dtype.name, data=value.tolist()))

def ndarray_deserialize(value):
    if value is None:
        return None
    np_array_obj = _NumpyArrayDTO(**value)
    return np.array(np_array_obj.data, dtype=np.dtype(np_array_obj.dtype))

NumpyField = lambda: field(
    metadata=config(
        encoder  = ndarray_serialize,
        decoder  = ndarray_deserialize,
        mm_field = marshmallow_numpy.NumpyField(),
    )
)
