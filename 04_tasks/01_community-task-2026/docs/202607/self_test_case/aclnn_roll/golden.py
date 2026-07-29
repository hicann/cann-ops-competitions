import torch
import numpy as np
from ml_dtypes import bfloat16


def calc_expect_func(x, shifts, dims):
    if isinstance(x, np.ndarray):
        shape = x.shape
        dtype = x.dtype
    else:
        shape = x.shape
        dtype = x.dtype

    if dtype == np.dtype('bfloat16'):
        x_tensor = torch.tensor(x.astype(np.float32)).to(torch.bfloat16)
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.to(torch.float32).numpy().astype(bfloat16)
    elif dtype == np.dtype('float16'):
        x_tensor = torch.tensor(x.astype(np.float32)).to(torch.float16)
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.to(torch.float32).numpy().astype(np.float16)
    elif dtype == np.dtype('float32'):
        x_tensor = torch.tensor(x.astype(np.float32))
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.numpy()
    elif dtype == np.dtype('float64'):
        x_tensor = torch.tensor(x.astype(np.float64))
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.numpy()
    elif dtype == np.dtype('complex64'):
        x_tensor = torch.tensor(x.astype(np.complex64))
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.numpy()
    elif dtype == np.dtype('complex128'):
        x_tensor = torch.tensor(x.astype(np.complex128))
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.numpy()
    elif dtype in [np.dtype('int32'), np.dtype('int64'), np.dtype('int16'), np.dtype('int8'), np.dtype('uint8'), np.dtype('uint32')]:
        x_tensor = torch.tensor(x)
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.numpy()
    elif dtype == np.dtype('bool'):
        x_tensor = torch.tensor(x.astype(np.uint8), dtype=torch.bool)
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.numpy().astype(bool)
    else:
        x_tensor = torch.tensor(x)
        result = torch.roll(x_tensor, shifts=shifts, dims=dims)
        result = result.numpy()

    return [result]
