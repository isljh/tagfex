import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


class SiBlurrySampler:
    """FlyGCL-style Si-Blurry task/session splitter.

    The splitter keeps the core protocol from FlyGCL's OnlineSampler but is
    intentionally dataset-agnostic: it only needs a target array and a class
    list, then exposes per-session sample indices. TAGFEX can train each
    session offline for many epochs using these indices.
    """

    def __init__(
        self,
        targets,
        classes,
        num_tasks=5,
        m=10,
        n=50,
        rnd_seed=1,
        rnd_NM=True,
    ):
        self.targets = np.asarray(targets)
        self.classes = [int(c) for c in classes]
        self.num_tasks = int(num_tasks)
        self.m = int(m)
        self.n = int(n)
        self.rnd_seed = int(rnd_seed)
        self.rnd_NM = bool(rnd_NM)
        self.generator = torch.Generator().manual_seed(self.rnd_seed)

        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive.")
        if len(self.classes) < self.num_tasks:
            raise ValueError(
                "Si-Blurry requires at least as many classes as tasks: "
                "{} classes, {} tasks.".format(len(self.classes), self.num_tasks)
            )

        if self.n == 100 or self.m == 0:
            self.n = 100
            self.m = 0

        self.disjoint_num = len(self.classes) * self.n // 100
        self.disjoint_num = int(self.disjoint_num // self.num_tasks) * self.num_tasks
        self.blurry_num = len(self.classes) - self.disjoint_num

        if self.rnd_NM:
            self._build_random_nm_split()
        else:
            self._build_even_split()

    def _class_order(self):
        perm = torch.randperm(len(self.classes), generator=self.generator).tolist()
        return [self.classes[i] for i in perm]

    def _build_even_split(self):
        class_order = self._class_order()
        disjoint_flat = class_order[: self.disjoint_num]
        blurry_flat = class_order[self.disjoint_num : self.disjoint_num + self.blurry_num]

        self.disjoint_classes = self._even_chunks(disjoint_flat)
        self.blurry_classes = self._even_chunks(blurry_flat)
        self._build_indices(equal_blur_split=True)

    def _build_random_nm_split(self):
        class_order = self._class_order()
        disjoint_flat = class_order[: self.disjoint_num]
        blurry_flat = class_order[self.disjoint_num : self.disjoint_num + self.blurry_num]

        self.disjoint_classes = self._random_chunks(disjoint_flat)
        self.blurry_classes = self._random_chunks(blurry_flat)
        self._build_indices(equal_blur_split=False)

    def _even_chunks(self, values):
        if len(values) == 0:
            return [[] for _ in range(self.num_tasks)]
        if len(values) % self.num_tasks != 0:
            raise ValueError(
                "Cannot evenly split {} classes across {} tasks. "
                "Use rnd_NM=True or choose compatible n/n_tasks.".format(
                    len(values), self.num_tasks
                )
            )
        width = len(values) // self.num_tasks
        return [values[i * width : (i + 1) * width] for i in range(self.num_tasks)]

    def _random_chunks(self, values):
        if len(values) == 0:
            return [[] for _ in range(self.num_tasks)]
        cuts = self._random_cut_points(len(values))
        return [values[cuts[i] : cuts[i + 1]] for i in range(self.num_tasks)]

    def _random_cut_points(self, total):
        if total <= 0:
            return [0] * (self.num_tasks + 1)
        cuts = torch.randint(
            0,
            total,
            (self.num_tasks - 1,),
            generator=self.generator,
        ).sort().values.tolist()
        return [0] + cuts + [total]

    def _build_indices(self, equal_blur_split):
        self.disjoint_indices = [[] for _ in range(self.num_tasks)]
        self.blurry_indices = [[] for _ in range(self.num_tasks)]

        disjoint_lookup = {
            cls: task_id
            for task_id, task_classes in enumerate(self.disjoint_classes)
            for cls in task_classes
        }
        blurry_lookup = {
            cls: task_id
            for task_id, task_classes in enumerate(self.blurry_classes)
            for cls in task_classes
        }

        for index, target in enumerate(self.targets):
            target = int(target)
            if target in disjoint_lookup:
                self.disjoint_indices[disjoint_lookup[target]].append(index)
            elif target in blurry_lookup:
                self.blurry_indices[blurry_lookup[target]].append(index)

        self._mix_blurry_indices(equal_blur_split)

        self.indices = [[] for _ in range(self.num_tasks)]
        for task_id in range(self.num_tasks):
            task_indices = self.disjoint_indices[task_id] + self.blurry_indices[task_id]
            if len(task_indices) > 0:
                order = torch.randperm(len(task_indices), generator=self.generator).tolist()
                task_indices = [task_indices[i] for i in order]
            self.indices[task_id] = task_indices

    def _mix_blurry_indices(self, equal_blur_split):
        blurred = []

        if equal_blur_split:
            for task_id in range(self.num_tasks):
                split_at = len(self.blurry_indices[task_id]) * self.m // 100
                blurred.extend(self.blurry_indices[task_id][:split_at])
                self.blurry_indices[task_id] = self.blurry_indices[task_id][split_at:]

            blurred = self._shuffle_indices(blurred)
            width = len(blurred) // self.num_tasks if self.num_tasks > 0 else 0
            for task_id in range(self.num_tasks):
                self.blurry_indices[task_id].extend(blurred[:width])
                blurred = blurred[width:]
            return

        total_blurry_samples = sum(len(x) for x in self.blurry_indices)
        total_to_blur = total_blurry_samples * self.m // 100
        if total_to_blur <= 0:
            return

        cuts = self._random_cut_points(total_to_blur)
        for task_id in range(self.num_tasks):
            take = cuts[task_id + 1] - cuts[task_id]
            blurred.extend(self.blurry_indices[task_id][:take])
            self.blurry_indices[task_id] = self.blurry_indices[task_id][take:]

        blurred = self._shuffle_indices(blurred)
        for task_id in range(self.num_tasks):
            take = cuts[task_id + 1] - cuts[task_id]
            self.blurry_indices[task_id].extend(blurred[:take])
            blurred = blurred[take:]

    def _shuffle_indices(self, indices):
        if len(indices) == 0:
            return []
        order = torch.randperm(len(indices), generator=self.generator).tolist()
        return [indices[i] for i in order]

    def set_task(self, task_id):
        if task_id < 0 or task_id >= self.num_tasks:
            raise ValueError("task_id out of range: {}".format(task_id))
        self.task = task_id

    def __iter__(self):
        task = getattr(self, "task", 0)
        return iter(self.indices[task])

    def __len__(self):
        task = getattr(self, "task", 0)
        return len(self.indices[task])
