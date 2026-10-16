"""兼容旧导入名称。当前公式对齐实现位于 paper_gn.py。

历史结果必须使用其 results*/source 内的原始模型重放，不能用当前模型加载。
"""
from paper_gn import EdgeModel, NodeModel, _Model
