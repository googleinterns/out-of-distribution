from typing import Any, Callable, Set, List

Hook = Callable[[List[float]], Any]   # the input list contains a value for each gradient-enabled parameter


class HookableOptimizer:
    learning_rates_hooks: Set[Hook]
    updates_to_weights_hooks: Set[Hook]
    regularization_to_gradients_hooks: Set[Hook]

    def __init__(self):
        self.learning_rates_hooks = set()
        self.updates_to_weights_hooks = set()
        self.regularization_to_gradients_hooks = set()
