import torch
import psutil, os
from fvcore.nn import FlopCountAnalysis
from model.m3_model import Net

def analyze_network_gpu_memory(model, inputs, device='cuda'):
    model = model.to(device).eval()
    # 清空并重置显存统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model(*[x.to(device) for x in inputs])

    # 峰值显存
    peak = torch.cuda.max_memory_allocated(device) / 1024**2  # 转成 MB
    print(f'Inference peak GPU memory: {peak:.2f} MB')

def analyze_network_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {params / 1e6}M')

def analyze_network_FLOPs(model):
    img = torch.randn(1, 3, 80, 80)
    geo = torch.randn(1, 2, 512)
    pts = torch.randn(1, 4, 512)
    flop_analyzer = FlopCountAnalysis(model, (img, geo, pts, img, geo, pts))
    total_flops = flop_analyzer.total()
    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")

def analyze_system_cpu_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # bytes
    return mem / 1024**2             # 转 MB

def analyze_system_gpu_memory():
    # 当前设备
    dev = torch.cuda.current_device()
    # 已分配给 tensors 的显存峰值
    alloc_peak = torch.cuda.max_memory_allocated(dev) # bytes
    # 已保留给缓存管理器的显存峰值（更接近实际占用）
    # reserved_peak = torch.cuda.max_memory_reserved(dev) # bytes

    return alloc_peak / 1024**2 # 转 MB


if __name__ == "__main__":
    # Initialize model
    # 构造一组典型输入，batch_size=1
    model = Net()
    x1 = torch.randn(1, 3, 80, 80)
    x2 = torch.randn(1, 2, 512)
    x3 = torch.randn(1, 4, 512)
    inputs = [x1, x2, x3, x1, x2, x3]  # 按 forward 签名放入

    analyze_network_gpu_memory(model, inputs)



