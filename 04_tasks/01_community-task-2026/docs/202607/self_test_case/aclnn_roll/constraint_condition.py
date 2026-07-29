import os
import random
import numpy as np


def constraint_condition(case):
    case["expect_func"] = "{}:calc_expect_func".format(os.path.abspath("golden.py"))

    for input_desc in case["input_desc"]:
        if input_desc["name"] == "x":
            dtype = input_desc["data_type"]
            if dtype in ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]:
                input_desc["value_range"] = [-100, 100]
            elif dtype == "bool":
                input_desc["value_range"] = [0, 1]
            elif dtype == "complex64":
                input_desc["value_range"] = [-10, 10]
            else:
                input_desc["value_range"] = [-10, 10]

    x_shape = case["input_desc"][0]["shape"]
    ndim = len(x_shape)

    for output_desc in case["output_desc"]:
        if output_desc["name"] == "out":
            output_desc["shape"] = list(x_shape)

    shifts_attr = None
    dims_attr = None
    for attr_desc in case["attr_desc"]:
        if attr_desc["name"] == "shifts":
            shifts_attr = attr_desc
        elif attr_desc["name"] == "dims":
            dims_attr = attr_desc

    if shifts_attr is None or dims_attr is None:
        return case

    use_empty_dims = random.choice([True, False])

    if use_empty_dims and ndim >= 1:
        dims_attr["value"] = []
        shifts_attr["value"] = [1]
    else:
        if ndim == 1:
            num_dims = 1
        else:
            num_dims = random.randint(1, min(ndim, 3))

        available_dims = list(range(-ndim, ndim))
        if num_dims <= len(available_dims):
            selected_dims = random.sample(available_dims, num_dims)
        else:
            selected_dims = random.choices(available_dims, k=num_dims)

        shifts_list = []
        for d in selected_dims:
            dim_size = x_shape[d] if d >= 0 else x_shape[d + ndim]
            if dim_size > 1:
                shift_val = random.randint(-dim_size * 2, dim_size * 2)
            else:
                shift_val = 0
            shifts_list.append(shift_val)

        dims_attr["value"] = selected_dims
        shifts_attr["value"] = shifts_list

    return case
