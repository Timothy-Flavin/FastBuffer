import torch
import numpy as np
import typing


class FastBuffer:
    def __init__(
        self,
        buffer_len: int,
        n_agents: int = 1,
        gpu: bool = False,
        action_mask: bool = False,
        discrete_cardonalities: typing.Iterable | None = None,
        registered_vars={
            "obs": {"dim": 12, "dtype": torch.float64, "per_agent": True},
            "obs_": {"dim": 12, "dtype": torch.float64, "per_agent": True},
            "reward": {"dim": 1, "dtype": torch.float64, "per_agent": False},
            "discrete_action": {"dim": 1, "dtype": torch.long, "per_agent": True},
            "continuous_action": {"dim": 2, "dtype": torch.float64, "per_agent": True},
        },
    ):
        self.buffer_len = buffer_len
        self.steps_recorded = 0
        self.current_idx = 0
        self.n_agents = n_agents
        self.gpu_enabled = gpu
        self.has_action_mask = action_mask
        self.discrete_cardonalites = discrete_cardonalities
        self.registered_vars = registered_vars
        self.current_cpu_idx = 0
        self.last_gpu_idx = 0
        # If the CPU has gone past buffer_len back to 0 this is set to true
        self.cpu_wrap_around = False
        # If the CPU has wrapped around and then passed up the last_gup_idx
        # then the whole buffer needs to get copied over
        self.full_buffer_refresh = False

    def setup_tensors(self):
        self.cpu_tensors = {}
        self.gpu_tensors = None
        if self.gpu_enabled:
            gpu_tensors = {}
        for k in self.registered_vars.keys():
            v = self.registered_vars[k]
            dim = []
            if v["per_agent"]:
                dim.append(self.n_agents)
            if v["dim"] is not None:
                if isinstance(v["dim"], int):
                    v["dim"] = [v["dim"]]
                if isinstance(v["dim"], typing.Iterable):
                    dim = dim + list(v["dim"])
