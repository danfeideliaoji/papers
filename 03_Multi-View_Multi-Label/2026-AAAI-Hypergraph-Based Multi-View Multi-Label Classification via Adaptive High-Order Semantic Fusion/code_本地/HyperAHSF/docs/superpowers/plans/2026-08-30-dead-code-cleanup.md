# 明确废代码清理计划

**目标：** 在不改变训练、推理和实验行为的前提下，删除会干扰阅读的旧实现与静态噪音。

## 清理边界

- 删除注释掉的历史语句、临时调试输出和 `return` 后的不可达实现。
- 删除未使用的导入、未读取的局部变量以及仓库内无引用的私有初始化方法。
- 保留数据预处理公共函数和可能由外部脚本调用的公开接口。
- 保留会影响模型 `state_dict` 结构及旧检查点兼容性的注册层，即使当前前向路径未使用。
- 不调整算法公式、张量形状、函数签名、分支条件或返回值。

## 执行步骤

1. 记录当前编译与可运行测试结果，作为行为基线。
2. 清理根目录训练、指标、预处理和模型代码。
3. 清理 `experiments/candidate1` 中的未使用代码与测试噪音。
4. 每批修改后运行 `python -m compileall -q .`。
5. 使用项目 CUDA 虚拟环境运行当前可执行测试，并记录因缺少依赖而无法执行的测试。

## 验证命令

```powershell
python -m compileall -q .
& 'E:\project\cuda\.venv\Scripts\python.exe' -m unittest `
  experiments.candidate1.tests.test_diagnostics `
  experiments.candidate1.tests.test_summary -v
```
