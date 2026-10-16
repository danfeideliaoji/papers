"""公共 HGNN 独立消融实验基础设施。"""

from .configuration import AblationFlags, validate_ablation_flags

__all__ = ["AblationFlags", "validate_ablation_flags"]
