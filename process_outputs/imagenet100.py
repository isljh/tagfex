
import os

import shutil

from tqdm import tqdm

# =======================

# 修改以下路径

# =======================

# 原始 ImageNet 训练集路径 (里面包含 n01514668 等子文件夹)

TRAIN_ROOT = r"E:\Downloads\ILSVRC2012_img_train"

# 整理后的输出路径

OUTPUT_ROOT = r"E:\continual-learning\datasets\ImageNet100\train"

# 对应的 train.txt 文件

TXT_FILE = r"E:\continual-learning\datasets\train.txt"

MODE = "copy"  # copy 或 symlink (建议用copy，防止路径变动导致软连接失效)


# =======================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_or_link(src, dst, mode="copy"):
    if os.path.exists(dst):
        return

    if mode == "copy":

        shutil.copy2(src, dst)

    elif mode == "symlink":

        os.symlink(src, dst)


def main():
    ensure_dir(OUTPUT_ROOT)

    current_class = None

    extracted = 0

    missing = 0

    # 读取 txt 获取所有行

    with open(TXT_FILE, "r") as f:

        lines = f.readlines()

    print(f"开始处理，共计 {len(lines)} 行...")

    for line in tqdm(lines, desc="提取进度"):

        line = line.strip()

        if not line:
            continue

        # 1. 处理类别行 (例如: n01514668)

        if "/" not in line:
            current_class = line

            ensure_dir(os.path.join(OUTPUT_ROOT, current_class))

            continue

        # 2. 处理图片行 (例如: n01514668/n01514668_23056.JPEG)

        assert current_class is not None

        # 【核心修改点】：在 train 数据集中，src_img 必须包含类别文件夹路径

        # 你的 txt 里的 line 已经是 "n01514668/n01514668_23056.JPEG"

        src_img = os.path.join(TRAIN_ROOT, line)

        # 目标路径保持一致

        dst_img = os.path.join(OUTPUT_ROOT, line)

        # 确保目标类别的文件夹存在

        ensure_dir(os.path.dirname(dst_img))

        if os.path.exists(src_img):

            copy_or_link(src_img, dst_img, MODE)

            extracted += 1

        else:

            # 有时 txt 里的路径斜杠方向不同，尝试处理一下

            alt_src = os.path.join(TRAIN_ROOT, line.replace('/', os.sep))

            if os.path.exists(alt_src):

                copy_or_link(alt_src, dst_img, MODE)

                extracted += 1

            else:

                # print(f"[MISSING] {src_img}")

                missing += 1

    print(f"\n处理完成！")

    print(f"成功提取: {extracted}")

    print(f"未能找到: {missing}")

    print(f"输出目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()