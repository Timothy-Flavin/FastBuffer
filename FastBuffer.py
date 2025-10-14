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
        self.gpu_steps_recorded = 0
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
            self.gpu_tensors = {}
        for k in self.registered_vars.keys():
            v = self.registered_vars[k]
            dim = []
            if v["per_agent"]:
                dim.append(self.n_agents)
            dim.append(self.buffer_len)
            if v["dim"] is not None:
                if isinstance(v["dim"], int):
                    v["dim"] = [v["dim"]]
                if isinstance(v["dim"], typing.Iterable):
                    dim = dim + list(v["dim"])
            print(f"Setting up tensor {k} with dim {dim} and dtype {v['dtype']}")
            self.cpu_tensors[k] = torch.zeros(*dim, dtype=v["dtype"])
            if self.gpu_enabled:
                self.gpu_tensors[k] = torch.zeros(*dim, dtype=v["dtype"]).cuda()

        if self.has_action_mask:
            self.masks = []
            self.next_masks = []
            for k in self.discrete_cardonalities:
                dim = self.n_agents, self.buffer_len, k
                self.masks.append(torch.ones(*dim, dtype=torch.bool))
                self.next_masks.append(torch.ones(*dim, dtype=torch.bool))
            self.cpu_tensors["action_mask"] = self.masks
            self.cpu_tensors["action_mask_"] = self.next_masks
            if self.gpu_enabled:
                self.gpu_tensors["action_mask"] = [
                    torch.ones_like(m).cuda() for m in self.masks
                ]
                self.gpu_tensors["action_mask_"] = [
                    torch.ones_like(m).cuda() for m in self.next_masks
                ]
        self.initialized = True

    def save_transition(self, data: dict):
        if not hasattr(self, "initialized"):
            self.setup_tensors()
        for k in data.keys():
            if k not in self.registered_vars.keys():
                raise KeyError(f"Key {k} not registered in buffer")
            v = self.registered_vars[k]
            if v["per_agent"]:
                self.cpu_tensors[k][:, self.current_cpu_idx] = torch.tensor(
                    data[k], dtype=v["dtype"]
                )
            else:
                self.cpu_tensors[k][self.current_cpu_idx] = torch.tensor(
                    data[k], dtype=v["dtype"]
                )
        self.current_cpu_idx += 1
        self.steps_recorded = min(self.steps_recorded + 1, self.buffer_len)
        if self.current_cpu_idx >= self.buffer_len:
            self.current_cpu_idx = 0
            self.cpu_wrap_around = True
        if (
            self.cpu_wrap_around
            and self.gpu_enabled
            and self.current_cpu_idx == self.last_gpu_idx
        ):
            self.full_buffer_refresh = True

    def update_gpu(self):
        if not self.gpu_enabled:
            raise RuntimeError("GPU not enabled for this buffer")
        if self.full_buffer_refresh:
            for k in self.registered_vars.keys():
                self.gpu_tensors[k][:] = self.cpu_tensors[k][:]
            if self.has_action_mask:
                for i in range(len(self.masks)):
                    self.gpu_tensors["action_mask"][i][:] = self.cpu_tensors[
                        "action_mask"
                    ][i][:]
                    self.gpu_tensors["action_mask_"][i][:] = self.cpu_tensors[
                        "action_mask_"
                    ][i][:]
            self.full_buffer_refresh = False
        else:
            if self.current_cpu_idx > self.last_gpu_idx:
                idxs = slice(self.last_gpu_idx, self.current_cpu_idx)
                gpu_idxs = idxs
            else:
                idxs = torch.concatenate(
                    (
                        torch.arange(self.last_gpu_idx, self.buffer_len),
                        torch.arange(0, self.current_cpu_idx),
                    )
                )
                gpu_idxs = idxs.detach().clone().cuda()
            for k in self.registered_vars.keys():
                if self.registered_vars[k]["per_agent"]:
                    self.gpu_tensors[k][:, gpu_idxs] = self.cpu_tensors[k][:, idxs].to(
                        self.gpu_tensors[k].device
                    )
                else:
                    self.gpu_tensors[k][gpu_idxs] = self.cpu_tensors[k][idxs].to(
                        self.gpu_tensors[k].device
                    )

            if self.has_action_mask:
                for i in range(len(self.masks)):
                    self.gpu_tensors["action_mask"][i][:, gpu_idxs] = self.cpu_tensors[
                        "action_mask"
                    ][i][:, idxs]
                    self.gpu_tensors["action_mask_"][i][:, gpu_idxs] = self.cpu_tensors[
                        "action_mask_"
                    ][i][:, idxs]
        self.last_gpu_idx = self.current_cpu_idx
        self.gpu_steps_recorded = self.steps_recorded

    def sample_transitions(self, batch_size, idxs=None):
        if self.gpu_enabled:
            tensors = self.gpu_tensors
        else:
            tensors = self.cpu_tensors

        last_idx = self.steps_recorded if not self.gpu_enabled else self.last_gpu_idx
        if idxs is None:
            idxs = np.random.randint(0, last_idx, size=batch_size)
        batch = {}
        for k in self.registered_vars.keys():
            v = self.registered_vars[k]
            if v["per_agent"]:
                batch[k] = tensors[k][:, idxs].clone().detach()
            else:
                batch[k] = tensors[k][idxs].clone().detach()
        if self.has_action_mask:
            batch["action_mask"] = [
                m[:, idxs].clone().detach() for m in tensors["action_mask"]
            ]
            batch["action_mask_"] = [
                m[:, idxs].clone().detach() for m in tensors["action_mask_"]
            ]
        return batch, idxs

    def reset(self):
        self.steps_recorded = 0
        self.gpu_steps_recorded = 0
        self.current_idx = 0
        self.current_cpu_idx = 0
        self.last_gpu_idx = 0
        self.cpu_wrap_around = False
        self.full_buffer_refresh = False
        for k in self.registered_vars.keys():
            self.cpu_tensors[k].zero_()
            if self.gpu_enabled:
                self.gpu_tensors[k].zero_()
        if self.has_action_mask:
            for i in range(len(self.masks)):
                self.masks[i].fill_(1)
                self.next_masks[i].fill_(1)
            if self.gpu_enabled:
                for i in range(len(self.masks)):
                    self.gpu_tensors["action_mask"][i].fill_(1)
                    self.gpu_tensors["action_mask_"][i].fill_(1)
