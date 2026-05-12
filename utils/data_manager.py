import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR10_AA, iCIFAR100_AA, iImageNet100_AA, \
    iImageNet100_LeJEPA, iCIFAR100_LeJEPA
from tqdm import tqdm
from utils.si_blurry_sampler import SiBlurrySampler
import torch

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, aug=1, args=None):
        self.dataset_name = dataset_name
        self.aug = aug
        self.args = args or {}
        self.use_si_blurry = _is_si_blurry(self.args)
        self._current_task = None
        self._setup_data(dataset_name, shuffle, seed, defer_class_mapping=self.use_si_blurry)
        if self.use_si_blurry:
            self._setup_si_blurry(seed)
            return
        assert init_cls <= len(self._class_order), "No enough classes."
        #_increments列表[10,10,10...]
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    # Number of Tasks
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_task_sizes(self):
        return list(self._increments)

    def set_current_task(self, task):
        self._current_task = task
        if self.use_si_blurry:
            self.train_sampler.set_task(task)
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        # appendent: 额外拼接进来的旧样本exemplar memory (old_data, old_targets)
        # indices：当前要取哪些类别
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        #flip模式常用于某些特征提取或测试时增强
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        if self.use_si_blurry and source == "train" and self._use_current_session(indices, ret_data):
            session_data, session_targets = self._get_si_blurry_session_data()
            data.append(session_data)
            targets.append(session_targets)
        elif self.use_si_blurry and source == "train" and ret_data:
            for idx in indices:
                class_data, class_targets = self._select_si_blurry_seen_class(idx)
                data.append(class_data)
                targets.append(class_targets)

        for idx in ([] if len(data) > 0 else indices):
            if m_rate is None:
                #data = [所有类0样本, 所有类1样本...] targets = [所有0标签, 所有1标签...]
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = _concat_data_targets(data, targets, self.use_path)
        #data = [所有类0样本, 所有类1样本...appendent_data] targets = [所有0标签, 所有1标签...appendent_targets]
        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path,self.aug if source == "train" and mode == "train" else 1)
        else:
            return DummyDataset(data, targets, trsf, self.use_path,self.aug if source == "train" and mode == "train" else 1)

    #为 finetune 专门做一个“比例受控”的数据集       
    def get_finetune_dataset(self,known_classes,total_classes,source,mode,appendent,type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx+1)
            num=len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot*(total_classes-known_classes)/known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot/(total_classes-known_classes))
        for idx in range(known_classes,total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data),new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data=np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path, self.aug if source == "train" and mode == "train" else 1)

    #训练集 + 验证集
    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path, self.aug
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed, defer_class_mapping=False):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data 格式numpy.ndarray
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order list：[0, 1, 2, ..., 99]
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)
        if defer_class_mapping:
            return

        # Map indices 重新映射标签
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def _setup_si_blurry(self, seed):
        n_tasks = int(_arg_or_default(self.args, "n_tasks", 5))
        n = int(_arg_or_default(self.args, "n", _arg_or_default(self.args, "disjoint_ratio", 50)))
        m = int(_arg_or_default(self.args, "m", _arg_or_default(self.args, "blurry_ratio", 10)))
        rnd_NM = bool(_arg_or_default(self.args, "rnd_NM", True))
        rnd_seed = int(_arg_or_default(self.args, "rnd_seed", seed))

        raw_train_targets = np.asarray(self._train_targets)
        raw_test_targets = np.asarray(self._test_targets)
        classes = sorted(np.unique(raw_train_targets).astype(int).tolist())
        self.train_sampler = SiBlurrySampler(
            raw_train_targets,
            classes,
            num_tasks=n_tasks,
            m=m,
            n=n,
            rnd_seed=rnd_seed,
            rnd_NM=rnd_NM,
        )

        exposure_order = []
        exposed = set()
        new_classes_by_task = []
        session_classes_by_task = []
        for task_indices in self.train_sampler.indices:
            task_new = []
            task_session_classes = []
            for sample_idx in task_indices:
                class_id = int(raw_train_targets[sample_idx])
                if class_id not in task_session_classes:
                    task_session_classes.append(class_id)
                if class_id not in exposed:
                    exposed.add(class_id)
                    exposure_order.append(class_id)
                    task_new.append(class_id)
            new_classes_by_task.append(task_new)
            session_classes_by_task.append(task_session_classes)

        for class_id in classes:
            if class_id not in exposed:
                exposure_order.append(class_id)

        self._class_order = exposure_order
        self._train_targets = _map_new_class_index(raw_train_targets, self._class_order)
        self._test_targets = _map_new_class_index(raw_test_targets, self._class_order)

        old_to_new = {old: new for new, old in enumerate(self._class_order)}
        self._si_blurry_new_classes = [
            [old_to_new[c] for c in task_classes] for task_classes in new_classes_by_task
        ]
        self._si_blurry_session_classes = [
            sorted([old_to_new[c] for c in task_classes])
            for task_classes in session_classes_by_task
        ]
        self._increments = [len(task_classes) for task_classes in self._si_blurry_new_classes]
        self._si_blurry_task_indices = [
            np.asarray(task_indices, dtype=np.int64) for task_indices in self.train_sampler.indices
        ]

        cumulative = []
        self._si_blurry_seen_indices = []
        for task_indices in self._si_blurry_task_indices:
            cumulative.extend(task_indices.tolist())
            self._si_blurry_seen_indices.append(np.asarray(cumulative, dtype=np.int64))

        logging.info(
            "Using Si-Blurry split: n_tasks={}, n={}, m={}, rnd_NM={}, rnd_seed={}".format(
                n_tasks, self.train_sampler.n, self.train_sampler.m, rnd_NM, rnd_seed
            )
        )
        for task_id in range(n_tasks):
            logging.info(
                "Si-Blurry task {} | disjoint(raw)={} | blurry(raw)={} | "
                "new(mapped)={} | session(mapped)={} | samples={}".format(
                    task_id,
                    self.train_sampler.disjoint_classes[task_id],
                    self.train_sampler.blurry_classes[task_id],
                    self._si_blurry_new_classes[task_id],
                    self._si_blurry_session_classes[task_id],
                    len(self._si_blurry_task_indices[task_id]),
                )
            )

    def _use_current_session(self, indices, ret_data):
        if self._current_task is None or ret_data:
            return False
        return len(indices) > 0

    def _get_si_blurry_session_data(self):
        task_indices = self._si_blurry_task_indices[self._current_task]
        return self._select_by_indices(self._train_data, self._train_targets, task_indices)

    def _select_si_blurry_seen_class(self, class_idx):
        if self._current_task is None:
            seen_indices = np.arange(len(self._train_targets))
        else:
            seen_indices = self._si_blurry_seen_indices[self._current_task]
        labels = self._train_targets[seen_indices]
        class_indices = seen_indices[np.where(labels == class_idx)[0]]
        return self._select_by_indices(self._train_data, self._train_targets, class_indices)

    def _select_by_indices(self, x, y, idxes):
        idxes = np.asarray(idxes, dtype=np.int64)
        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = [x[int(idx)] for idx in idxes]
        return x_return, y[idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, aug=1):
        assert len(images) == len(labels), "Data size error!"
        self.aug = aug
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.aug == 1:
            if self.use_path:
                image = self.trsf(pil_loader(self.images[idx]))
            else:
                image = self.trsf(Image.fromarray(self.images[idx]))
            label = self.labels[idx]
            return idx, image, label
        #idx, aug_view1, aug_view2, label
        else:
            if self.use_path:
                images = [self.trsf(pil_loader(self.images[idx])) for _ in range(self.aug)]
            else:
                images = [self.trsf(Image.fromarray(self.images[idx])) for _ in range(self.aug)]
            label = self.labels[idx]
            return idx, *images, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _is_si_blurry(args):
    setting = str(_arg_or_default(args, "setting", _arg_or_default(args, "data_protocol", ""))).lower()
    si_blurry = _arg_or_default(args, "si_blurry", False)
    if isinstance(si_blurry, str):
        si_blurry = si_blurry.lower() in {"1", "true", "yes", "y"}
    return bool(si_blurry) or setting in {"si_blurry", "flygcl"}


def _arg_or_default(args, key, default):
    value = args.get(key, default)
    return default if value is None else value


def _concat_data_targets(data, targets, use_path):
    data = [d for d in data if len(d) > 0]
    targets = [t for t in targets if len(t) > 0]
    if len(data) == 0:
        empty_data = np.asarray([]) if use_path else np.empty((0,))
        return empty_data, np.asarray([], dtype=np.int64)
    if use_path:
        merged_data = []
        for item in data:
            if isinstance(item, np.ndarray):
                merged_data.extend(item.tolist())
            else:
                merged_data.extend(item)
        merged_data = np.asarray(merged_data)
    else:
        merged_data = np.concatenate(data)
    return merged_data, np.concatenate(targets)


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "cifar100_lejepa":
        return iCIFAR100_LeJEPA()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar100_aa":
        return iCIFAR100_AA()
    elif name == "cifar10_aa":
        return iCIFAR10_AA()
    elif name == "imagenet100_aa":  # 添加这一行
        return iImageNet100_AA()
    elif name == "imagenet100_lejepa":  # 添加这一行
        return iImageNet100_LeJEPA()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
