import torch
import matplotlib.pyplot as plt
import numpy as np


def analyze_tagfex_weights(checkpoint_path):
    # 1. 加载权重
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # 2. 自动定位 fc.weight (处理 DDP 包装的情况)
    if 'fc.weight' in state_dict:
        weights = state_dict['fc.weight']
    elif 'module.fc.weight' in state_dict:
        weights = state_dict['module.fc.weight']
    else:
        # 打印出所有键名帮你排查，如果上面两个都不对
        print("未找到默认键名，当前权重文件中的键有：", state_dict.keys())
        return

    # 3. 计算每个类别的 L2 Norm
    # weights 的 shape 是 [num_classes, feature_dim]
    norms = torch.norm(weights, p=2, dim=1).numpy()
    num_classes = len(norms)

    # 4. 开始绘图
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
    plt.bar(range(num_classes), norms, color=colors, alpha=0.8)

    # 标记任务边界 (ImageNet100 初始通常是 10 类)
    plt.axvline(x=9.5, color='red', linestyle='--', label='Initial Task Boundary')

    plt.title(f"Class Weight Norm Distribution (Total Classes: {num_classes})", fontsize=14)
    plt.xlabel("Class Index", fontsize=12)
    plt.ylabel("L2 Norm of Weights", fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend()

    save_path = "weight_norm_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"分析完成！图片已保存至: {save_path}")
    plt.show()


# 使用你的路径
path = "logs/tagfex/imagenet100_lejepa/0/10/20260202_171458/checkpoints/reproduce_1993_task_0.pth"
analyze_tagfex_weights(path)