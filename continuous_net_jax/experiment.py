from datetime import datetime
import json
import os

import flax.linen as nn
from flax.training import checkpoints

from .tools import module_to_dict, module_to_single_line, load_model_dict_from_json


class Experiment():
    """Manages saving and loading checkpoints and descriptions."""
    model: nn.Module
    path: str

    def __init__(self, model=None, path=None, scope=None):
        if model is None:
            self.path = path
            self.model = load_model_dict_from_json(
                os.path.join(self.path, 'model.json'), scope)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = os.path.join(
                path, f"{module_to_single_line(model)}_{timestamp}")
            self.model = model
            self.save_model_description()

    def save_model_description(self):
        try:
            os.makedirs(self.path)
        except FileExistsError:
            print(f"In danger of overwriting {self.path}/")
        with open(os.path.join(self.path, "model_repr.txt"), "w") as f:
            f.write(repr(self.model))
        with open(os.path.join(self.path, "model.json"), "w") as f:
            f.write(json.dumps(module_to_dict(self.model), indent=4))

    def save_optimizer_hyper_params(self, optimizer, seed):
        h_dict = {"seed": seed}
        h_dict.update(optimizer.hyper_params.__dict__)
        with open(os.path.join(self.path, "optimizer_hyper_params.txt"),
                  "w") as f:
            f.write(f"{repr(optimizer.hyper_params)}\nseed = {seed}")
        with open(os.path.join(self.path, "optimizer_hyper_params.json"),
                  "w") as f:
            f.write(json.dumps(h_dict, indent=4))

    def save_checkpoint(self, optimizer, step):
        checkpoints.save_checkpoint(self.path, optimizer, step=step, keep=3)

    def load_checkpoint(self, like=None):
        return checkpoints.restore_checkpoint(self.path, like)
