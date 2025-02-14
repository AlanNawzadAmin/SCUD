import torch

def root_finder(func, x0, x1, ts, max_iter=100, tol=1e-12):
    """ Variation on Brentq """
    def secant_step(x0, x1, f0, f1):
        return x1 - f1 * (x1 - x0) / (f1 - f0 + 1e-15)  # Add small epsilon to avoid division by zero
    def bisect_step(x0, x1, f0, f1):
        return (x1 + x0) / 2
    
    x0 = x0 * torch.ones_like(ts)
    x1 = x1 * torch.ones_like(ts)
    x2 = x1.clone()
    
    f0 = func(x0, ts)
    f1 = func(x1, ts)
    f2 = f1.clone()

    bisect_mode = torch.ones_like(ts).bool()
    mask = torch.ones_like(ts).bool()

    for _ in range(max_iter):
        x2[mask] = torch.where(bisect_mode,
                         bisect_step(x0, x1, f0, f1),
                         secant_step(x0, x1, f0, f1))[mask]
        f2[mask] = func(x2[mask], ts[mask])
        bisect_mode = torch.where(torch.abs(f2) < 0.5 * torch.abs(f1 - f0), # did well
                                  bisect_mode,
                                  ~bisect_mode)

        update_x1 = torch.sign(f2) == torch.sign(f1)
        x0[mask] = torch.where(update_x1, x0, x1)[mask]
        x1[mask] = x2[mask]
        f0[mask] = torch.where(update_x1, f0, f1)[mask]
        f1[mask] = f2[mask]

        mask = torch.abs(f2) > tol
        if (~mask).all():
            break
    
    return torch.where(torch.abs(f2) < tol, x2, torch.full_like(x2, float('nan')))

def newton_root_finder(func, x0, ts, min_x=torch.tensor(1e-8), max_iter=1000, tol=1e-12, print_=False):
    ts = ts.detach()
    x = x0 * torch.ones_like(ts).double()
    x = torch.maximum(x, min_x)
    f = torch.ones_like(ts)
    df = torch.ones_like(ts)
    mask = torch.ones_like(ts).bool()

    for _ in range(max_iter):
        with torch.enable_grad():
            x_masked = x[mask].detach().requires_grad_(True)
            f_masked = func(x_masked, ts[mask])
            df_masked = torch.autograd.grad(f_masked.sum(), x_masked)[0]
            f[mask] = f_masked.detach()
            df[mask] = df_masked  

        step =  - f / df
        # step = torch.sign(step) * torch.clamp(torch.abs(step), None, 3*torch.abs(x)/4)
        x_at_min = (x == min_x) * (step < 0)
        x[mask] = torch.where(mask, x + step, x)[mask]
        x = torch.maximum(x, min_x)

        mask = (torch.abs(f) > tol) * (~x_at_min)
        if (~mask).all():
            break
        if print_:
            print("av errs", (torch.abs(f)[torch.abs(f) > tol]).mean(),
                  "N:", (torch.abs(f) > tol).sum())

    return x#torch.where(torch.abs(f) < tol, x, torch.full_like(x, float('nan')))
