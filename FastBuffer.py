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
        self.discrete_cardonalities = discrete_cardonalities
        self.registered_vars = registered_vars
        self.current_cpu_idx = 0
        self.last_gpu_idx = 0
        # If the CPU has gone past buffer_len back to 0 this is set to true
        self.cpu_wrap_around = False
        # If the CPU has wrapped around and then passed up the last_gup_idx
        # then the whole buffer needs to get copied over
        self.full_buffer_refresh = False

        self.copy_stream = None
        self.copy_event = None
        if self.gpu_enabled:
            self.copy_stream = torch.cuda.Stream()
            self.copy_event = torch.cuda.Event()

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
            # Allocate CPU tensors. When GPU is enabled, allocate CPU
            # tensors with pinned memory to allow non-blocking (async)
            # transfers to the device.
            if self.gpu_enabled:
                assert self.gpu_tensors is not None
                gt = self.gpu_tensors
                self.cpu_tensors[k] = torch.zeros(
                    *dim, dtype=v["dtype"], pin_memory=True
                )
                gt[k] = torch.zeros(*dim, dtype=v["dtype"]).cuda()
            else:
                self.cpu_tensors[k] = torch.zeros(*dim, dtype=v["dtype"])

        if self.has_action_mask and self.discrete_cardonalities is not None:
            self.masks = []
            self.next_masks = []
            for k in self.discrete_cardonalities:
                dim = self.n_agents, self.buffer_len, k
                self.masks.append(torch.ones(*dim, dtype=torch.bool))
                self.next_masks.append(torch.ones(*dim, dtype=torch.bool))
            self.cpu_tensors["action_mask"] = self.masks
            self.cpu_tensors["action_mask_"] = self.next_masks
            if self.gpu_enabled:
                assert self.gpu_tensors is not None
                gt = self.gpu_tensors
                gt["action_mask"] = [torch.ones_like(m).cuda() for m in self.masks]
                gt["action_mask_"] = [
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
                self.cpu_tensors[k][:, self.current_cpu_idx] = data[k]
            else:
                self.cpu_tensors[k][self.current_cpu_idx] = data[k]

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
        # These are initialized in __init__ when gpu is enabled
        assert self.copy_stream is not None
        assert self.copy_event is not None
        assert self.gpu_tensors is not None

        # Perform all device copies on the dedicated copy stream
        with torch.cuda.stream(self.copy_stream):  # type: ignore[arg-type]
            gt = self.gpu_tensors  # local alias for type checkers
            if self.full_buffer_refresh:
                for k in self.registered_vars.keys():
                    gt[k][:] = (
                        self.cpu_tensors[k][:]
                        .detach()
                        .clone()
                        .to(gt[k].device, non_blocking=True)
                    )
                if self.has_action_mask:
                    for i in range(len(self.masks)):
                        gt["action_mask"][i][:] = (
                            self.cpu_tensors["action_mask"][i][:]
                            .detach()
                            .clone()
                            .to(gt["action_mask"][i].device, non_blocking=True)
                        )
                        gt["action_mask_"][i][:] = (
                            self.cpu_tensors["action_mask_"][i][:]
                            .detach()
                            .clone()
                            .to(gt["action_mask_"][i].device, non_blocking=True)
                        )
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
                        gt[k][:, gpu_idxs] = (
                            self.cpu_tensors[k][:, idxs]
                            .detach()
                            .to(gt[k].device, non_blocking=True)
                        )
                    else:
                        gt[k][gpu_idxs] = (
                            self.cpu_tensors[k][idxs]
                            .detach()
                            .to(gt[k].device, non_blocking=True)
                        )

                if self.has_action_mask:
                    for i in range(len(self.masks)):
                        gt["action_mask"][i][:, gpu_idxs] = (
                            self.cpu_tensors["action_mask"][i][:, idxs]
                            .detach()
                            .to(gt["action_mask"][i].device, non_blocking=True)
                        )
                        gt["action_mask_"][i][:, gpu_idxs] = (
                            self.cpu_tensors["action_mask_"][i][:, idxs]
                            .detach()
                            .to(gt["action_mask_"][i].device, non_blocking=True)
                        )

        # Record event so compute stream can wait before reading
        self.copy_event.record(self.copy_stream)  # type: ignore[arg-type]
        self.last_gpu_idx = self.current_cpu_idx
        self.gpu_steps_recorded = self.steps_recorded

    def sample_transitions(self, batch_size, idxs=None):
        if self.gpu_enabled:
            tensors = self.gpu_tensors
        else:
            tensors = self.cpu_tensors
        assert tensors is not None, "Tensors not initialized"
        steps_recorded = (
            self.steps_recorded if not self.gpu_enabled else self.gpu_steps_recorded
        )
        if steps_recorded < batch_size:
            raise ValueError(
                f"Not enough steps recorded to sample {batch_size} transitions"
            )
        if idxs is None:
            # generate indices on the device to avoid host->device index copy
            device = tensors[list(self.registered_vars.keys())[0]].device
            idxs = torch.randint(
                0, steps_recorded, size=(batch_size,), device=device, dtype=torch.long
            )
        else:
            # Ensure idxs is a long tensor on the correct device
            if not isinstance(idxs, torch.Tensor):
                device = tensors[list(self.registered_vars.keys())[0]].device
                idxs = torch.as_tensor(idxs, dtype=torch.long, device=device)
            else:
                device = tensors[list(self.registered_vars.keys())[0]].device
                idxs = idxs.to(device=device, dtype=torch.long)

        batch = {}
        for k in self.registered_vars.keys():
            v = self.registered_vars[k]
            if v["per_agent"]:
                # per_agent tensors have shape [n_agents, buffer_len, ...]
                # use index_select on dim=1 to avoid advanced indexing overhead
                batch[k] = tensors[k].index_select(1, idxs).contiguous()
            else:
                # shape [buffer_len, ...], select on dim=0
                batch[k] = tensors[k].index_select(0, idxs).contiguous()

        if self.has_action_mask:
            batch["action_mask"] = [
                m.index_select(1, idxs).contiguous() for m in tensors["action_mask"]
            ]
            batch["action_mask_"] = [
                m.index_select(1, idxs).contiguous() for m in tensors["action_mask_"]
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
                assert self.gpu_tensors is not None
                self.gpu_tensors[k].zero_()
        if self.has_action_mask:
            for i in range(len(self.masks)):
                self.masks[i].fill_(1)
                self.next_masks[i].fill_(1)
            if self.gpu_enabled:
                assert self.gpu_tensors is not None
                for i in range(len(self.masks)):
                    self.gpu_tensors["action_mask"][i].fill_(1)
                    self.gpu_tensors["action_mask_"][i].fill_(1)
