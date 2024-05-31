import torch

def f(x, y):
    try:
        x = x - y
    except:
        return x
    return y

opt_f = torch.compile(f, backend="eager")
inp_x = torch.tensor([1,0,1,0], dtype=torch.bool)
inp_y = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
opt_f(inp_x, inp_y)
