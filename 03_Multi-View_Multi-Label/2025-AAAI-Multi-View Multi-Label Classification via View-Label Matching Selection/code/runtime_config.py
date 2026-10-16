"""训练可重复性设置：必须在首次 CUDA 矩阵运算前设置 cuBLAS 环境。"""
import os

# 两种工作区配置均被 PyTorch 支持；保留用户已设置的有效配置。
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
    raise RuntimeError('CUBLAS_WORKSPACE_CONFIG 请使用 :4096:8 或 :16:8')


def configure_determinism():
    import torch
    # 不采用 warn_only：遇到无法确定性执行的算子就报错，防止悄悄退回原训练。
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def runtime_metadata():
    import torch
    return {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
