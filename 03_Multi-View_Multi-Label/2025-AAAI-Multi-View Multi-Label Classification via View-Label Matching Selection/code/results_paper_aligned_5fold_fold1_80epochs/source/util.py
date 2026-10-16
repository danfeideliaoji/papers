"""修正版训练接口；旧实现完整保存在 results/source 中供历史结果重放。"""
from strict_training import run


def k_fold(args, data, target):
    # 统一转入严格训练/验证/测试流程，避免旧接口继续用测试折选模。
    return run(args, data, target)
