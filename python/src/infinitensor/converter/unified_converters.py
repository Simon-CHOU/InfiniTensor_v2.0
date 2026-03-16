import torch.nn as nn
from .registry import registry

#https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml

@registry.register("matmul","default")
def convert_matmul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.gemm(a, b, None)

@registry.register("add","Tensor")
def convert_add(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.add(a, b, None)

@registry.register("mul","Tensor")
def convert_mul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.mul(a, b, None)

@registry.register("sub","Tensor")
def convert_sub(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.sub(a, b, None)

@registry.register("clamp","default")
def convert_clip(translator, node):
    import torch
    from pyinfinitensor import ShapeExpr, dtype_from_string
    
    input_tensor = translator.tensors[node.args[0]]
    
    def get_or_create_tensor(val, name_suffix):
        if isinstance(val, torch.fx.Node):
            return translator.tensors[val]
        else:
            # It's a scalar or constant, create a tensor
            t_val = torch.tensor([val], dtype=torch.float32)
            # keep reference to prevent GC
            if not hasattr(translator, "constant_tensors"):
                translator.constant_tensors = []
            translator.constant_tensors.append(t_val)
            
            dtype = dtype_from_string(str(t_val.dtype))
            inf_tensor = translator.builder.tensor(ShapeExpr(list(t_val.shape)), dtype)
            inf_tensor.set_data(t_val.data_ptr(), translator.runtime)
            return inf_tensor

    min_val = node.args[1] if len(node.args) > 1 else node.kwargs.get('min')
    max_val = node.args[2] if len(node.args) > 2 else node.kwargs.get('max')
    
    min_tensor = get_or_create_tensor(min_val, "min")
    max_tensor = get_or_create_tensor(max_val, "max")
    
    translator.tensors[node] = translator.builder.clip(input_tensor, min_tensor, max_tensor, None)