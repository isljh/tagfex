import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader
import os
import json

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


def verify_gaussian_distribution(checkpoint_path, json_path, data_manager, save_name="embedding.png", device="cuda"):
    print(f"读取配置: {json_path}")
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    args = DummyArgs(config_dict)
    actual_dev = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu"
    args["device"] = [actual_dev]

    network = TagFexNet(args, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=actual_dev)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    network.load_state_dict(new_state_dict, strict=False)
    network.to(actual_dev)
    network.eval()

    # 加载数据
    init_cls = args.get('init_cls', 10)
    dataset = data_manager.get_dataset(np.arange(0, init_cls), source="test", mode="test")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("正在提取 embedding...")
    embeddings = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs = data[1].to(actual_dev)
            if imgs.dim() == 5:
                imgs = imgs[:, 0, :, :, :]  # 只取第一个视图 [N, C, H, W]

            # 直接跑 ta_net + projector，和训练时完全一致
            ta_fmap = network.ta_net(imgs)['fmaps'][-1]                      # [N, 512, H, W]
            ta_feature = ta_fmap.flatten(2).permute(0, 2, 1).mean(1)        # [N, 512]
            embedding = network.projector(ta_feature)                         # [N, 1024]

            embeddings.append(embedding.cpu().numpy())
            if i >= 20:
                break

    all_feats = np.concatenate(embeddings).flatten()
    print(f"采集完成：{len(all_feats)} 个特征点。范围: [{all_feats.min():.4f}, {all_feats.max():.4f}]")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(all_feats, bins=150, density=True, color='indianred', alpha=0.6, label='Embedding Distribution')

    mu, std = norm.fit(all_feats)
    x = np.linspace(all_feats.min(), all_feats.max(), 300)
    plt.plot(x, norm.pdf(x, mu, std), 'b--', linewidth=2, label=f'Fit (mu={mu:.2f}, std={std:.2f})')

    # 标准高斯作为目标参考
    x_std = np.linspace(-5, 5, 300)
    plt.plot(x_std, norm.pdf(x_std, 0, 1), 'g-', linewidth=2, label='Standard Gaussian (mu=0, std=1)')

    plt.axvline(x=0, color='black', linewidth=1)
    plt.title("Embedding Distribution vs Standard Gaussian")
    plt.legend()
    plt.savefig(save_name, dpi=300)
    print(f"✅ 图片已保存至 {save_name}")


if __name__ == "__main__":
    JSON_FILE = "exps/tagfex.json"
    dm = DataManager("imagenet100_aa", True, 1993, 10, 10, 1)
    """
    # 加了 SIGReg 的 checkpoint
    verify_gaussian_distribution(
        checkpoint_path="logs/tagfex/imagenet100_lejepa/0/10/20260202_171458/checkpoints/reproduce_1993_task_0.pth",
        json_path=JSON_FILE,
        data_manager=dm,
        save_name="embedding_with_sigreg.png"
    )
    """
    # 没加 SIGReg 的 checkpoint（改成你实际的路径）
    verify_gaussian_distribution(
        checkpoint_path = "logs/tagfex/imagenet100_aa/0/10/checkpoints/reproduce_1993_task_0.pth",
        #checkpoint_path="logs/tagfex/imagenet100_lejepa/0/10/20260304_100347/checkpoints/reproduce_1993_task_0.pth",
        json_path=JSON_FILE,
        data_manager=dm,
        save_name="embedding_no_sigreg1.png"
    )
