import torch
import numpy as np
from ml_dtypes import bfloat16


def calc_expect_func(x1, x2):
    if x1.dtype == np.dtype('bfloat16'):
        x1_tensor = torch.tensor(x1.astype(np.float32)).to(torch.bfloat16)
        x2_tensor = torch.tensor(x2.astype(np.float32)).to(torch.bfloat16)
        result = torch.remainder(x1_tensor, x2_tensor)
        result = result.to(torch.float32).numpy().astype(bfloat16)
    elif x1.dtype == np.dtype('float16'):
        x1_tensor = torch.tensor(x1.astype(np.float32)).to(torch.float16)
        x2_tensor = torch.tensor(x2.astype(np.float32)).to(torch.float16)
        result = torch.remainder(x1_tensor, x2_tensor)
        result = result.to(torch.float32).numpy().astype(np.float16)
    else:
        x1_tensor = torch.tensor(x1)
        x2_tensor = torch.tensor(x2)
        result = torch.remainder(x1_tensor, x2_tensor)
        result = result.numpy()

    return [result]
