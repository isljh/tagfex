import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 加在最顶部，所有import之前
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader
import json
from torch import nn
from convs.linears import TagFex_SimpleLinear

from utils.inc_net import TagFexNet
from utils.data_manager import DataManager


class DummyArgs(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return self if name == "config" else None

    def get(self, key, default=None):
        return super().get(key, default)


def extract_pre_relu(checkpoint_path, json_path, data_manager,
                     label="model",
                     proj_output_dim=1024,
                     use_simclr_projector=False,
                     device="cuda"):
    print(f"\n[{label}] 读取配置: {json_path}")
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    args = DummyArgs(config_dict)
    actual_dev = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu"
    args["device"] = [actual_dev]
    args["proj_output_dim"] = proj_output_dim

    network = TagFexNet(args, pretrained=False)

    if use_simclr_projector:
        ta_feature_dim = network.ta_feature_dim
        proj_hidden_dim = args.get("proj_hidden_dim", 2048)
        network.projector = nn.Sequential(
            TagFex_SimpleLinear(ta_feature_dim, proj_hidden_dim),
            nn.ReLU(True),
            TagFex_SimpleLinear(proj_hidden_dim, proj_output_dim),
        ).to(actual_dev)

    state_dict = torch.load(checkpoint_path, map_location=actual_dev)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    network.load_state_dict(new_state_dict, strict=False)
    network.to(actual_dev)
    network.eval()

    # 注册 Hook 在 layer4[-1].bn2，即最后一个 ReLU 之前
    raw_activations = []
    def hook_fn(module, input, output):
        raw_activations.append(output.detach().cpu())

    handle = network.ta_net.layer4[-1].bn2.register_forward_hook(hook_fn)

    init_cls = args.get('init_cls', 10)
    dataset = data_manager.get_dataset(np.arange(0, init_cls), source="test", mode="test")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    print(f"[{label}] 正在提取 Pre-ReLU 特征...")
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs = data[1].to(actual_dev)
            if imgs.dim() == 5:
                imgs = imgs[:, 0, :, :, :]
            _ = network.ta_net(imgs)
            if i >= 20:
                break

    handle.remove()

    feats = []
    for act in raw_activations:
        # [N, 512, H, W] → GAP → [N, 512] → flatten
        gap = act.mean(dim=(2, 3)).flatten().numpy()
        feats.append(gap)

    all_feats = np.concatenate(feats)
    print(f"[{label}] 采集完成：{len(all_feats)} 个特征点。范围: [{all_feats.min():.4f}, {all_feats.max():.4f}]")
    return all_feats


def plot_all(results, save_name="pre_relu_comparison.png"):
    """
    并排子图：每个模型一张，风格参考 LeJEPA notebook：
      - 直方图 (bins=50, density=True, alpha=0.5)
      - 叠加高斯拟合曲线
      - 叠加标准高斯曲线作为参考
      - 横轴限制在 (-1, 1)
    """
    colors = ['indianred', 'steelblue', 'mediumseagreen']
    n = len(results)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    x_std = np.linspace(-1, 1, 300)
    std_gaussian = norm.pdf(x_std, 0, 1)

    for ax, (label, all_feats), color in zip(axes, results, colors):
        # 裁剪到 (-1, 1) 范围内再拟合，与 LeJEPA notebook xlim 对齐
        mu, std = norm.fit(all_feats)  # 全量数据拟合
        clipped = all_feats[(all_feats >= -1) & (all_feats <= 1)]  # 只用于画直方图显示

        # 直方图
        ax.hist(clipped, bins=50, density=True, alpha=0.5,
                color=color, label=f'{label}\n(μ={mu:.2f}, σ={std:.2f})')

        # 拟合高斯曲线
        x_fit = np.linspace(-1, 1, 300)
        ax.plot(x_fit, norm.pdf(x_fit, mu, std),
                color=color, linewidth=2, linestyle='--', label='Fitted Gaussian')

        # 标准高斯参考线
        ax.plot(x_std, std_gaussian,
                'k-', linewidth=1.5, linestyle='-', label='Standard Gaussian\n(μ=0, σ=1)')

        ax.axvline(x=0, color='black', linewidth=0.8, linestyle=':')
        ax.set_xlim(-1, 1)
        ax.set_title(f'{label} Distribution', fontsize=13)
        ax.set_xlabel('Feature Value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Pre-ReLU Feature Distribution vs Standard Gaussian", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图片已保存至 {save_name}")


if __name__ == "__main__":
    print("开始初始化 DataManager...")
    JSON_FILE = "exps/tagfex.json"
    dm = DataManager("imagenet100_aa", True, 1993, 10, 10, 1)
    print("DataManager 初始化完成")

    results = []

    # 原版 SimCLR
    feats = extract_pre_relu(
        #checkpoint_path = "E:/PyCIL_DDP/logs/tagfex/imagenet100_aa/0/10/checkpoints/reproduce_1993_task_0.pth",
        checkpoint_path="logs/tagfex/imagenet100_aa/0/10/checkpoints/reproduce_1993_task_0.pth",
        json_path=JSON_FILE,
        data_manager=dm,
        label="SimCLR",
        proj_output_dim=1024,
        use_simclr_projector=True,
    )
    results.append(("SimCLR", feats))

    # LeJEPA + SIGReg
    feats = extract_pre_relu(
        checkpoint_path="logs/tagfex/imagenet100_lejepa/0/10/20260306_100658/checkpoints/reproduce_1993_task_0.pth",
        #checkpoint_path="logs/tagfex/imagenet100_lejepa/0/10/20260304_100347/checkpoints/reproduce_1993_task_0.pth",
        json_path=JSON_FILE,
        data_manager=dm,
        label="LeJEPA + SIGReg",
        proj_output_dim=1024,
    )
    results.append(("LeJEPA + SIGReg", feats))

    plot_all(results, save_name="pre_relu_comparison5.png")