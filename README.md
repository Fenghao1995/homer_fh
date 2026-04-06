# FailCorrector (FIPER Integration Edition)

> 说明：当前执行环境无法直接访问 GitHub（对 `github.com`/`raw.githubusercontent.com` 返回 403），
> 因此无法在本仓库内直接拉取并改写 `utiasDSL/fiper` 源码。当前版本提供可直接嵌入 FIPER 的桥接实现与可运行测试。

## 当前目录放置

`failcorrector/` 与 `tests/` 位于项目主目录下（repo root）。
这是 Python 工程常见布局，便于后续复制到 FIPER 根目录或作为子模块引入。

## 新增/升级模块

- `failcorrector/core.py`：闭环纠偏主控制器（触发、纠偏、DreamZero 未来重评估、fallback）
- `failcorrector/fiper_bridge.py`：**无须改 FIPER 方法名** 的桥接器（通过 `FiperMethodMap` 绑定）
- `failcorrector/interfaces.py`：接口协议
- `failcorrector/config.py`：阈值、趋势门控、损失权重
- `failcorrector/example_usage.py`：桥接集成可运行示例

## 标记说明

所有创新点都带：

- `### [FailCorrector-NEW]`

## 如何融入真实 FIPER（推荐）

1. 将 `failcorrector/` 复制到 FIPER 根目录；
2. 在 FIPER 推理入口中实例化 `FiperBridgeController`；
3. 用 `FiperMethodMap` 映射 FIPER 中真实方法名；
4. 将每个 control step 的 `obs_history` 传入 `step(...)`，执行 `corrected_actions`。

## 快速运行

```bash
python -m failcorrector.example_usage
pytest -q
```
