import cv2
import numpy as np
import os
import torch
import glob

from depth_anything_v2.dpt import DepthAnythingV2

# 设备选择
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 配置模型
encoder_type = 'vitb'  # 可选: 'vits', 'vitb', 'vitl', 'vitg'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# 加载模型
depth_anything = DepthAnythingV2(**model_configs[encoder_type])
checkpoint_path = f'/media/disk/Depth-Anything-V2-main/depth_anything_v2_{encoder_type}.pth'
depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

# 输入输出路径
input_path = r"/media/disk/ultralytics-main/datasets/fire/train/images_shenduwanzheng/"
output_dir = r"/media/disk/ultralytics-main/datasets/fire/train/images_depth_vitb/"
os.makedirs(output_dir, exist_ok=True)

# 获取所有输入图片
image_files = glob.glob(os.path.join(input_path, "*.png")) + \
              glob.glob(os.path.join(input_path, "*.jpg")) + \
              glob.glob(os.path.join(input_path, "*.jpeg")) + \
              glob.glob(os.path.join(input_path, "*.png")) + \
              glob.glob(os.path.join(input_path, "*.JPG")) + \
              glob.glob(os.path.join(input_path, "*.jpeg"))

# 获取已保存的深度图（基于文件名）
existing_depth_files = set(os.path.basename(f) for f in glob.glob(os.path.join(output_dir, "*.png")))

# 过滤掉已经存储的图片
missing_files = [f for f in image_files if os.path.basename(f).replace('.jpg', '.png').replace('.jpeg', '.png') not in existing_depth_files]

print(f"共 {len(image_files)} 张图片，缺失 {len(missing_files)} 张，需要重新处理")

# 仅处理缺失的图片
input_size = 518  # 预测输入尺寸

for i, img_path in enumerate(missing_files):
    print(f"正在处理 {i + 1}/{len(missing_files)}: {img_path}")

    # 读取图像
    raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if raw_image is None:
        print(f"⚠️ 无法读取 {img_path}，跳过")
        continue

    # 进行深度预测
    depth = depth_anything.infer_image(raw_image, input_size)

    # 归一化深度图到 0-255
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    # 组合 RGB 和 深度图，形成 4 通道图像
    rgba_image = np.dstack((raw_image, depth_normalized))

    # 保存路径
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".png")

    try:
        cv2.imwrite(output_path, rgba_image)
        print(f"✅ 已保存: {output_path}")
    except Exception as e:
        print(f"❌ 存储失败 {output_path}: {e}")

print("所有缺失的 RGBD 深度图已处理完毕")
