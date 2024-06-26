import torch

def kde(x, std = 0.1, max_num_cmp = 20_000):
    # use a gaussian kernel to estimate density
    
    if len(x.shape) != 2:
        raise ValueError(f"Needs shape N, D got shape {x.shape}")
    x = x.half() # Do it in half precision TODO: remove hardcoding
    inds = torch.multinomial(torch.ones_like(x[:,0]), min(max_num_cmp, x.shape[-2]), replacement=False)
    y = x[inds]
    scores = (-torch.cdist(x,y)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density