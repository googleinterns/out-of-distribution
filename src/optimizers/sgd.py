import torch
from torch import optim

from src.optimizers.hookable_optimizer import HookableOptimizer


class SGD(optim.SGD, HookableOptimizer):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        optim.SGD.__init__(
            self, params, lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov
        )
        HookableOptimizer.__init__(self)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        learning_rates = []
        updates_to_weights = []
        regularization_to_gradients = []

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # log the regularization-to-gradient ratio
                regularization_norm = weight_decay * torch.norm(p).item()
                gradient_norm = torch.norm(p.grad).item()
                ratio = 1e6 if gradient_norm == 0 else regularization_norm / gradient_norm
                regularization_to_gradients.append(ratio)

                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # log the learning rate
                learning_rates.append(group["lr"])

                # log the update-to-weight ratio
                update_norm = group["lr"] * torch.norm(d_p).item()
                weight_norm = torch.norm(p).item()
                ratio = 1e6 if weight_norm == 0 else update_norm / weight_norm
                updates_to_weights.append(ratio)

                p.add_(d_p, alpha=-group["lr"])

        for hook in self.learning_rates_hooks:
            hook(learning_rates)
        for hook in self.regularization_to_gradients_hooks:
            hook(regularization_to_gradients)
        for hook in self.updates_to_weights_hooks:
            hook(updates_to_weights)

        return loss
