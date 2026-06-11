import torch
import torch.distributed as dist
from dual_language_models.optimizers.muon_utils import COEFF_LIST, muon_update, adam_update, hyperball_update



class SingleDeviceMuon(torch.optim.Optimizer):
    
    def __init__(self, param_groups, ns_steps: int = 5, coeffs: str = "jordan", kimi_adjust_lr: bool = False, ratio_adjust_lr: bool = False, normuon: bool = False, polar_express: bool = False, hyperball: bool = False):
        for group in param_groups:
            if group.get("use_muon", False):
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["coeff_list"] = group.get("coeff_list", COEFF_LIST[coeffs])
                group["ns_steps"] = group.get("ns_steps", ns_steps)
                group["lr_mul"] = group.get("lr_mul", 1)
                assert {"params", "lr", "lr_mul", "momentum", "beta2", "weight_decay", "coeff_list", "ns_steps", "use_muon"} <= set(group.keys())
            elif group.get("use_adamh", False):
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert {"params", "lr", "betas", "eps", "weight_decay", "use_adamh"} <= set(group.keys())
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert {"params", "lr", "betas", "eps", "weight_decay"} <= set(group.keys())
        super().__init__(param_groups, dict())
        self.normuon = normuon
        self.kimi_adjust_lr = kimi_adjust_lr
        self.ratio_adjust_lr = ratio_adjust_lr
        self.polar_express = polar_express
        self.hyperball = hyperball

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        state["R"] = p.norm()
                        if self.kimi_adjust_lr:
                            state["lr_adjust"] = 0.2 * max(p.size(-1), p.size(-2))**0.5
                        elif self.ratio_adjust_lr:
                            state["lr_adjust"] = (max(p.size(-2), p.size(-1)) / min(p.size(-2), p.size(-1)))**0.5
                        else:
                            state["lr_adjust"] = max(1, p.size(-2) / p.size(-1))**0.5
                    update = muon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                                         beta=group["momentum"], beta2=group["beta2"], normuon=self.normuon,
                                         polar=self.polar_express, coeff_list=group["coeff_list"], ns_steps=group["ns_steps"])
                    eff_lr = group["lr"] * state["lr_adjust"] * group["lr_mul"]
                    if self.hyperball:
                        update = hyperball_update(p, update, state["R"], eff_lr)
                    else:
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-eff_lr)
            elif group["use_adamh"]:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                        state["R"] = p.norm()
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    hyperball_update(p, update, state["R"], group["lr"])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class DistributedMuon(torch.optim.Optimizer):
    
    def __init__(self, param_groups, ns_steps: int = 5, coeffs: str = "jordan", kimi_adjust_lr: bool = False, ratio_adjust_lr: bool = False, normuon: bool = False, polar_express: bool = False, hyperball: bool = False):
        for group in param_groups:
            if group.get("use_muon", False):
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["eps"] = group.get("eps", 1e-10)
                group["coeff_list"] = group.get("coeff_list", COEFF_LIST[coeffs])
                group["ns_steps"] = group.get("ns_steps", ns_steps)
                group["lr_mul"] = group.get("lr_mul", 1)
                assert {"params", "lr", "lr_mul", "momentum", "beta2", "weight_decay", "coeff_list", "ns_steps", "use_muon"} <= set(group.keys())
            elif group.get("use_adamh", False):
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert {"params", "lr", "betas", "eps", "weight_decay", "use_adamh"} <= set(group.keys())
            else:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert {"params", "lr", "betas", "eps", "weight_decay"} <= set(group.keys())
        super().__init__(param_groups, dict())
        self.normuon = normuon
        self.kimi_adjust_lr = kimi_adjust_lr
        self.ratio_adjust_lr = ratio_adjust_lr
        self.polar_express = polar_express
        self.hyperball = hyperball

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                            state["R"] = p.norm()
                            if self.kimi_adjust_lr:
                                state["lr_adjust"] = 0.2 * max(p.size(-1), p.size(-2))**0.5
                            elif self.ratio_adjust_lr:
                                state["lr_adjust"] = (max(p.size(-2), p.size(-1)) / min(p.size(-2), p.size(-1)))**0.5
                            else:
                                state["lr_adjust"] = max(1, p.size(-2) / p.size(-1))**0.5
                        update = muon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                                            beta=group["momentum"], beta2=group["beta2"], normuon=self.normuon,
                                            polar=self.polar_express, coeff_list=group["coeff_list"], ns_steps=group["ns_steps"])
                        eff_lr = group["lr"] * state["lr_adjust"] * group["lr_mul"]
                        if self.hyperball:
                            update = hyperball_update(p, update, state["R"], eff_lr)
                        else:
                            if group["weight_decay"] and had_grad:
                                p.mul_(1 - group["lr"] * group["weight_decay"])
                            p.add_(update.reshape(p.shape), alpha=-eff_lr)
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            elif group["use_adamh"]:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                        state["R"] = p.norm()
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    hyperball_update(p, update, state["R"], group["lr"])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss