import torch
from ultralytics.nn.tasks import DetectionModel  # 请确保你信任该模型来源

# 定义路径
model_path = "/media/disk/ultralytics-main/yolo11n.pt"
new_model_path = "/media/disk/ultralytics-main/yolo11n_4ch.pt"

# 安全加载模型（允许 DetectionModel 全局变量）
with torch.serialization.safe_globals([DetectionModel]):
    model = torch.load(model_path, map_location='cpu', weights_only=False)

# 获取模型的第一层（自定义Conv模块），该模块包含conv, bn, act三个部分
first_layer = model['model'].model[0]
print("Original first layer:", first_layer)

# 获取内部卷积层
old_conv = first_layer.conv
old_weights = old_conv.weight.detach().clone()  # 原始权重 shape: [out_channels, 3, kernel, kernel]

# 创建新的卷积层，in_channels=4，其余参数保持不变
new_conv = torch.nn.Conv2d(
    in_channels=4,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
)

# 创建新的权重张量，并将前3个通道赋值为原权重，第4个通道初始化为0
new_weights = torch.zeros((old_conv.out_channels, 4, *old_conv.weight.shape[2:]), device=old_weights.device)
new_weights[:, :3, :, :] = old_weights
new_conv.weight = torch.nn.Parameter(new_weights)
if old_conv.bias is not None:
    new_conv.bias = old_conv.bias

# 替换 first_layer 中的 conv 模块
first_layer.conv = new_conv

# 如果模型中有 YAML 配置，更新其中的通道数为4
if hasattr(model['model'], 'yaml') and isinstance(model['model'].yaml, dict):
    model['model'].yaml['ch'] = 4

# 保存修改后的模型
torch.save(model, new_model_path)
print(f"Modified model saved at: {new_model_path}")
