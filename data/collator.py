import collections
import re

import numpy as np
from torch._six import string_classes

"""
    data collator for a torch dataloader
    to be used with jax and co
"""

"""
    refactored the pytorch dataloader default collate function to work with jax
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
"""


# dataset returns and instace as a dict
def data_collator(batch):
    pass


np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


# torch returns everything as a torch.tensor
# since that's incompatible with jax,
# convert to numpy
def data_collator_fn(batch):
    # let's repurpose the base collate_fn from pytorch
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, np.ndarray):
        out = None
        return np.stack(batch, 0, out=out)

    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))

            return data_collator_fn([np.array(b) for b in batch])
        elif elem.shape == ():  # scalars
            return np.array(batch)
    elif isinstance(elem, float):
        return np.array(batch, dtype=np.float32)
    elif isinstance(elem, int):
        return np.array(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):  # type: ignore
        try:
            return elem_type({key: data_collator_fn([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: data_collator_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(data_collator_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):  # type: ignore
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        # It may be accessed twice, so we use a list.
        transposed = list(zip(*batch))

        if isinstance(elem, tuple):
            # Backwards compatibility.
            return [data_collator_fn(samples) for samples in transposed]
        else:
            try:
                return elem_type([data_collator_fn(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [data_collator_fn(samples) for samples in transposed]
