# bambooML 现代化改进说明

本文档说明了对 bambooML 框架的现代化改进，参考了 Made-With-ML 框架的最佳实践。

## 主要改进

### 1. CLI 框架升级：从 argparse 迁移到 Typer

**改进前：**
- 使用传统的 `argparse` 库
- CLI 代码冗长，缺少类型提示
- 帮助信息不够友好

**改进后：**
- 使用现代化的 `Typer` 框架
- 完整的类型注解和自动帮助生成
- 更好的 CLI 体验和错误提示

**示例：**
```bash
# 改进后的 CLI 使用方式保持不变，但体验更好
bambooml train --help  # 现在有更好的帮助信息
```

### 2. 实验跟踪：集成 MLflow

**新增功能：**
- 添加了 `bambooml/core/experiment.py` 模块
- 支持实验跟踪、参数记录、指标记录和模型注册
- 可选依赖，不影响现有功能

**使用方法：**
```bash
# 安装 MLflow 支持
pip install -e ".[monitor]"

# 训练时指定实验名称
bambooml train --experiment-name "my_experiment" ...
```

**功能：**
- 自动记录超参数
- 记录每个 epoch 的验证损失
- 保存模型检查点
- 支持模型注册和版本管理

### 3. 增强的评估模块

**改进前：**
- 只有基础的 `accuracy` 和 `mse` 函数
- 缺少分类任务的完整指标

**改进后：**
- 添加了 `precision`, `recall`, `f1` 等指标
- 新增 `get_classification_metrics` 函数，提供全面的分类指标
- 新增 `evaluate.py` 模块，提供统一的评估接口
- 支持整体指标和每个类别的详细指标

**示例：**
```python
from bambooml.tasks.evaluate import evaluate_model

metrics = evaluate_model(y_true, y_pred, class_names=["class1", "class2"])
# 返回包含 overall 和 per_class 指标的字典
```

### 4. 改进的日志系统

**改进前：**
- 简单的日志配置
- 只支持控制台和单个文件输出

**改进后：**
- 支持日志轮转（RotatingFileHandler）
- 分离 info 和 error 日志文件
- 更详细的日志格式（包含文件名、函数名、行号）
- 更好的日志级别管理

**配置：**
- Info 日志：`logs/info.log`（10MB 轮转，保留 10 个备份）
- Error 日志：`logs/error.log`（10MB 轮转，保留 10 个备份）

### 5. 测试框架

**新增内容：**
- 完整的 pytest 测试结构
- 基础测试用例（metrics, config）
- pytest 配置（`pyproject.toml`）
- 测试覆盖率支持

**运行测试：**
```bash
# 运行所有测试
make test

# 运行测试并生成覆盖率报告
make test-cov
```

### 6. 代码质量工具

**新增配置：**
- `black`：代码格式化
- `isort`：导入排序
- `flake8`：代码检查
- `pre-commit`：Git hooks

**使用方法：**
```bash
# 安装开发依赖
make install-dev

# 格式化代码
make style

# 清理临时文件
make clean
```

### 7. 项目配置改进

**pyproject.toml：**
- 添加了开发依赖组
- 配置了代码格式化工具（black, isort）
- 配置了 pytest 和覆盖率
- 更好的依赖管理

**requirements.txt：**
- 创建了标准的 requirements.txt
- 清晰标注了可选依赖

**Makefile：**
- 添加了常用的开发命令
- 统一了代码风格和质量检查流程

## 向后兼容性

所有改进都保持了向后兼容性：
- CLI 命令和参数保持不变
- 现有代码无需修改即可使用
- MLflow 和测试框架都是可选的

## 迁移指南

### 对于现有用户

1. **更新依赖：**
   ```bash
   pip install -e ".[dev]"  # 如果需要开发工具
   pip install -e ".[monitor]"  # 如果需要 MLflow
   ```

2. **使用新的 CLI（可选）：**
   - CLI 命令保持不变，但体验更好
   - 可以开始使用 `--experiment-name` 参数进行实验跟踪

3. **使用新的评估功能：**
   ```python
   from bambooml.tasks.evaluate import evaluate_model
   # 替代原来的简单指标计算
   ```

### 对于开发者

1. **设置开发环境：**
   ```bash
   make install-dev
   ```

2. **运行测试：**
   ```bash
   make test
   ```

3. **提交代码前：**
   ```bash
   make style  # 自动格式化
   ```

## 未来改进方向

参考 Made-With-ML，可以考虑的进一步改进：

1. **分布式训练：** 集成 Ray Train（类似 Made-With-ML）
2. **模型服务：** 添加 FastAPI + Ray Serve 支持
3. **CI/CD：** 添加 GitHub Actions 工作流
4. **文档：** 使用 mkdocs 生成文档网站
5. **数据验证：** 集成 Great Expectations 进行数据质量检查

## 总结

这些改进使 bambooML 更加现代化和易用，同时保持了其轻量级和专注于 HEP 数据处理的特色。所有改进都遵循了软件工程最佳实践，提高了代码质量和可维护性。

