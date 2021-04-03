from datetime import datetime
import json
import os

import flax.linen as nn
from flax.training import checkpoints

from .tools import *


class Experiment():
    """Manages saving and loading checkpoints and descriptions."""
    model: nn.Module
    path: str

    def __init__(self, model=None, path=None, scope=None):
        if model is None:
            self.path = path
            self.model = load_model_dict_from_json(
                os.path.join(self.path, 'model.json'), scope)
            with open(self._path('optimizer_hyper_params.json'), "r") as f:
                h_dict = json.loads(f.read())
            self.seed = h_dict.pop('seed')
            self.extra = h_dict.pop('extra')
            self.optimizer_def = parse_optimizer_def_dict(h_dict, scope)
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = os.path.join(
                path, f"{module_to_single_line(model)}_{timestamp}")
            self.model = model
            self.save_model_description()

    def _path(self, fname: str) -> str:
        return os.path.join(self.path, fname)

    def save_model_description(self):
        try:
            os.makedirs(self.path)
        except FileExistsError:
            print(f"In danger of overwriting {self.path}/")
        print("Saving a model to ", self.path)
        with open(self._path("model_repr.txt"), "w") as f:
            f.write(repr(self.model))
        with open(self._path("model.json"), "w") as f:
            f.write(json.dumps(module_to_dict(self.model), indent=4))

    def save_optimizer_hyper_params(self, optimizer_def, seed, extra=None):
        h_dict = optimizer_def_to_dict(optimizer_def)
        h_dict["seed"] = seed
        if extra:
            h_dict["extra"] = extra
        with open(self._path("optimizer_hyper_params.txt"), "w") as f:
            f.write(f"{repr(optimizer_def.hyper_params)}\nseed = {seed}")
            if extra:
                f.write(f"\nextra = {seed}")
        with open(self._path("optimizer_hyper_params.json"), "w") as f:
            f.write(json.dumps(h_dict, indent=4))

    def save_checkpoint(self, optimizer, state, step):
        checkpoints.save_checkpoint(self.path, {'optimizer': optimizer, 'state':state}, step=step, keep=3)

    def load_checkpoint(self, like=None):
        return checkpoints.restore_checkpoint(self.path, like)
