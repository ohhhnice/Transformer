from modelscope.hub import snapshot_download

# 模型名称，格式为"模型作者/模型名称"
model_id = "Qwen/Qwen3-4B"

# 指定下载路径
save_dir = "./model/qwen3_4b/"

# 下载模型
model_dir = snapshot_download(
    model_id=model_id,
    cache_dir=save_dir,
    # 可选参数：是否强制重新下载
    # force_download=True
)

print(f"模型已下载至：{model_dir}")