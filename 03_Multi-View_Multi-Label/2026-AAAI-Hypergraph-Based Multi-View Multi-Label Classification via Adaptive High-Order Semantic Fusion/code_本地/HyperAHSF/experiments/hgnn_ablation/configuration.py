"""三种 HGNN 单机制消融的公共开关与互斥校验。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationFlags:
    """描述一次运行启用的单一机制；三个开关默认全部关闭。"""

    use_residual: bool = False
    learnable_hyperedge_weight: bool = False
    use_attention: bool = False

    @property
    def enabled_count(self) -> int:
        """返回当前启用的机制数量。"""

        return sum(
            map(
                int,
                (
                    self.use_residual,
                    self.learnable_hyperedge_weight,
                    self.use_attention,
                ),
            )
        )


def validate_ablation_flags(flags: AblationFlags) -> None:
    """拒绝同时启用两种或三种机制的组合实验。"""

    if flags.enabled_count > 1:
        raise ValueError("三种机制只能单独启用，不能同时启用两个或三个机制。")
