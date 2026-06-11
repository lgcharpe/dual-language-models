from __future__ import annotations

from itertools import repeat

import torch
from dual_language_models.optimizers.polar_express import optimal_composition

COEFF_LIST = {
    "jordan": [(3.4445, -4.7750,  2.0315)],
    "polar_five_iter": optimal_composition(l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.02),
    "polar_default": optimal_composition(l=1e-3, num_iters=10, safety_factor_eps=1e-2, cushion=0.02),
}

# Original code from https://github.com/NoahAmsel/PolarExpress/blob/main/polar_express.py
@torch.compile
def polar_express(G: torch.Tensor, steps: int, coeff_list: list[tuple[float, float, float]] = COEFF_LIST["polar_default"]) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = coeff_list[:steps] + list( 
        repeat(coeff_list[-1], steps - len(coeff_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# copied from https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, coeff_list=COEFF_LIST["jordan"]):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = coeff_list[0]  
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# Original code from https://github.com/zichongli5/NorMuon/blob/main/normuon.py
def muon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True, normuon=False, polar=False, coeff_list=COEFF_LIST["jordan"]):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    if polar:
        update = polar_express(update, steps=ns_steps, coeff_list=coeff_list)
    else:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps, coeff_list=coeff_list)
    update = update.to(grad.dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)
    ################ NorMuon added ###################
    if normuon:
        vnorm = update.norm(dim=(-2,-1), keepdim=True)
        v_mean = torch.mean(update * update, dim=-1, keepdim=True)
        second_momentum.lerp_(v_mean, 1 - beta2)
        step_size = 1 / second_momentum.sqrt().add_(1e-10)
        update.mul_(step_size)
        vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
        update.mul_(vnorm / (vnorm_new.add_(1e-10))) # This scaling keep the update norm the same as pre-normalization
    ##################################################
    # update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


def hyperball_update(p, u, R, lr, eps=1e-10):
    u.mul_(R / torch.clamp(u.norm(), min=eps))
    p.add_(u, alpha=-lr)
    p.mul_(R / torch.clamp(p.norm(), min=eps))