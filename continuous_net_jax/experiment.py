import json
import os

import flax.linen as nn
from flax.training import checkpoints

from .tools import module_to_dict


class Experiment():
    name: str
    model: nn.Module

    def __init__(self, name, model=None):
        self.name = name
        if model is None:
            self.model = self.load_model_description()
        else:
            self.model = model
            self.save_model_description()

    def save_model_description(self):
        try:
            os.mkdir(self.name)
        except FileExistsError:
            print(f"In danger of overwriting {self.name}/")
        with open(f"{self.name}/model_repr.txt", "w") as f:
            f.write(repr(self.model))
        with open(f"{self.name}/model.json", "w") as f:
            f.write(json.dumps(module_to_dict(self.model), indent=4))

    def load_model_description(self):
        with open(f"{self.name}/model.json", "r") as f:
            model_dict = json.loads(f.read())
        raise NotImplemented

    def save_checkpoint(self, optimizer, step):
        checkpoints.save_checkpoint(self.name, optimizer, step=step, keep=3)

    def load_checkpoint(self, like):
        return checkpoints.restore_checkpoint(self.name, like)
