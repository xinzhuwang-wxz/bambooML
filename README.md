# bambooML

bambooML 是一个面向高能物理（HEP）与通用 ML/LLM 的现代化、轻量化、可扩展框架。它复用 weaver 的“配置驱动 + 按需处理”思想，通过一个统一的 CLI 在本地或集群上批量提交训练、推理、导出等任务。

## 主要特性
- 配置驱动数据处理（对齐 weaver）：
  - `selection/test_time_selection/new_variables/inputs/labels/observers/weights`
  - 自动标准化（`center:auto`）、裁剪（`min/max`）、填充（`length/pad_mode`）、分组堆叠为张量
  - 依赖解析（AST）与表达式求值，按需构造新变量
  - 多源读取：`.root/.h5/.parquet/.csv`，支持 `treename/branch_magic/file_magic`
- 任务子系统：分类/回归最小训练闭环、推理输出、ONNX 导出
- LLM 微调：`llm-finetune`（LoRA 入口，可扩展训练循环）
- 监控与调试：TensorBoard 集成、数据检查 `data-inspect`
- 批量与提交：`submit` 生成本地/SLURM 作业脚本
- **现代化改进（v0.1.0+）：**
  - 🎯 使用 Typer 的现代化 CLI 框架
  - 📊 MLflow 实验跟踪和模型注册（可选）
  - 📈 增强的评估指标（precision, recall, F1, 分类报告等）
  - 📝 改进的日志系统（日志轮转、分离 info/error）
  - 🧪 完整的测试框架（pytest）
  - 🔧 代码质量工具（black, isort, flake8, pre-commit）

## 安装
- 获取代码（以 GitHub 为例）：

```
git clone https://github.com/xinzhuwang-wxz/bambooML.git
cd bambooML
```
- 新建环境
```
conda create -n bambooML python=3.10
conda activate bambooML
```

- 安装（普通用户与开发者二选一）：
  - 普通安装：
    - ```pip install .```
  - 开发者可编辑安装（修改源码立即生效），或者组合：
    - ```pip install -e .```
    - ```pip install -e .[hep,monitor,llm]```  # 包含可选依赖
    - ```pip install -e .[dev]```  # 开发工具（black, pytest 等）
- 可选依赖（按需安装）：
  - HEP 数据支持：```pip install -e .[hep]``` 或 ```pip install uproot tables```
  - 实验跟踪：```pip install -e .[monitor]``` 或 ```pip install mlflow tensorboard```
  - LLM 微调：```pip install -e .[llm]``` 或 ```pip install transformers peft accelerate```

- 验证 CLI：

```
bambooml --help
```

## 快速开始
- 训练（示例 CSV）：

```
bambooml train \
  -c examples/data.yaml \
  -n examples/model.py \
  -i _:examples/data.csv \
  --num-epochs 2 --batch-size 2 \
  --tensorboard runs/exp1
```

- 推理：

```
bambooml predict \
  -c examples/data.yaml \
  -n examples/model.py \
  -m checkpoints/*runs/exp1/network_best_epoch_state.pt \
  -t _:examples/data.csv \
  --predict-output output.parquet
```

- 导出 ONNX：

```
bambooml export \
  -c examples/data.yaml \
  -n examples/model.py \
  -m checkpoints/*runs/exp1/network_best_epoch_state.pt \
  --export-onnx export/model.onnx
```

（以上示例均可把 `examples/...` 替换为你自己的数据与模型路径）

- 数据检查：

```
bambooml data-inspect -c bambooML/examples/data.yaml -i _:bambooML/examples/data.csv
```

- 生成提交脚本（本地/SLURM）：

```
bambooml submit \
  --system slurm \
  --script bambooML/examples/train_job.sh \
  --cmdline "bambooml train -c bambooML/examples/data.yaml -n bambooML/examples/model.py -i _:bambooML/examples/data.csv --num-epochs 1 --batch-size 2"
```

## 数据配置要点（对齐 weaver）
- `inputs`：定义输入分组与变量，以及 `length/pad_mode/center/scale/min/max/pad_value`
- `new_variables`：以表达式形式定义新变量，按需自动解析依赖并计算
- `selection/test_time_selection`：选择表达式；选择前自动构造缺失变量
- `labels`：简单多类或自定义字典；自动生成 `_label_/_labelcheck_` 做一致性检查
- `weights`：可选二维直方图 reweight 与采样均衡

## 监控与训练增强
- `--tensorboard`：记录验证损失等标量到 TensorBoard
- `--use-amp`：启用混合精度（AMP）
- `--lr-scheduler flat+decay`：学习率调度（平稳 + 衰减）
- 多卡：传入多卡列表时自动使用 DataParallel
- 快照目录：默认根目录为 `checkpoints`，并按 `checkpoints/<时间_主机>runs/<实验名>/` 分层（训练逻辑见 `bambooml/tasks/train.py:137-146,160-167`）

## 与 weaver 的关系
- 复用其“配置驱动的数据处理”范式，并以轻量方式实现：
  - AST 提名与表达式求值、自动标准化/裁剪/填充及分组堆叠
  - 多源文件读取与 `treename/branch_magic/file_magic` 支持
  - 迭代数据加载与异步预取（简化版本）
- 训练/推理/导出流程保持与 weaver 接近的接口风格，便于迁移与学习

## 目录结构
- `bambooml/`：顶层包
  - `core/`：配置、日志、注册表
  - `data/`：DataConfig、IO、预处理、Dataset、工具
  - `tasks/`：训练、推理、导出、数据检查
  - `llm/`：LoRA 入口
  - `monitor/`：TensorBoard 集成
  - `runner/`：CLI、提交脚本生成
- `examples/`：示例 `data.yaml/model.py/data.csv`

## 现代化改进

bambooML v0.1.0+ 引入了多项现代化改进，参考了 Made-With-ML 框架的最佳实践：

- ✅ **Typer CLI**：现代化的命令行接口，更好的帮助信息和类型提示
- ✅ **MLflow 集成**：实验跟踪、参数记录、指标记录和模型注册
- ✅ **增强评估**：完整的分类指标（precision, recall, F1, 分类报告）
- ✅ **改进日志**：日志轮转、分离 info/error、更详细的格式
- ✅ **测试框架**：pytest 测试结构和基础测试用例
- ✅ **代码质量**：black, isort, flake8, pre-commit hooks

详细改进说明请参阅 [IMPROVEMENTS.md](IMPROVEMENTS.md)

### 使用 MLflow 实验跟踪

```bash
# 安装 MLflow 支持
pip install -e ".[monitor]"

# 训练时指定实验名称
bambooml train \
  --experiment-name "my_experiment" \
  -c examples/data.yaml \
  -n examples/model.py \
  ...
```

### 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
make test

# 运行测试并生成覆盖率报告
make test-cov
```

## 常见问题
- 未安装时可用 `python -m bambooml.runner.cli` 代替 `bambooml`
- ROOT/HDF5 文件需要安装相应依赖（如 `uproot/tables`）
- LoRA 需安装 `transformers/peft/accelerate`
- MLflow 是可选的，不影响核心功能
- 开发 vs 安装：`pip install -e .` 为可编辑安装，修改源码后无需重装；若修改了 `pyproject.toml` 的入口或依赖，请重新执行安装

## 参考
- weaver：https://github.com/hqucms/weaver-core.git
- Made-With-ML：https://github.com/GokuMohandas/Made-With-ML（现代化改进的参考框架）