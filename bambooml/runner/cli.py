"""Modern CLI using Typer."""
import glob
from typing import List, Optional

import typer
from typing_extensions import Annotated

from ..tasks.train import run as run_train
from ..tasks.predict import run as run_predict
from ..tasks.export import run as run_export
from ..llm.lora import run as run_lora

app = typer.Typer(help="bambooML: Lightweight ML/LLM framework")


def parse_filespecs(specs: List[str]) -> dict:
    """Parse file specifications into a dictionary.

    Args:
        specs: List of file specifications in format "name:path" or just "path".

    Returns:
        Dictionary mapping names to file lists.
    """
    file_dict = {}
    for f in specs:
        if ':' in f:
            name, fp = f.split(':', 1)
        else:
            name, fp = '_', f
        files = glob.glob(fp)
        file_dict[name] = sorted(files)
    return file_dict


@app.command()
def train(
    data_config: Annotated[str, typer.Option("-c", "--data-config", help="Path to data configuration YAML")],
    network_config: Annotated[str, typer.Option("-n", "--network-config", help="Path to network configuration Python file")],
    data_train: Annotated[List[str], typer.Option("-i", "--data-train", help="Training data files")] = [],
    data_val: Annotated[List[str], typer.Option("-l", "--data-val", help="Validation data files")] = [],
    extra_selection: Annotated[Optional[str], typer.Option("--extra-selection", help="Extra selection expression")] = None,
    data_fraction: Annotated[float, typer.Option("--data-fraction", help="Fraction of data to use")] = 1.0,
    file_fraction: Annotated[float, typer.Option("--file-fraction", help="Fraction of files to use")] = 1.0,
    fetch_by_files: Annotated[bool, typer.Option("--fetch-by-files", help="Fetch data by files")] = False,
    fetch_step: Annotated[float, typer.Option("--fetch-step", help="Fetch step size")] = 0.01,
    in_memory: Annotated[bool, typer.Option("--in-memory", help="Load data in memory")] = False,
    train_val_split: Annotated[float, typer.Option("--train-val-split", help="Train/val split ratio")] = 0.8,
    no_remake_weights: Annotated[bool, typer.Option("--no-remake-weights", help="Don't remake weights")] = False,
    log: Annotated[Optional[str], typer.Option("--log", help="Log file path")] = None,
    network_option: Annotated[List[str], typer.Option("-o", "--network-option", help="Network options (key value pairs)")] = [],
    num_epochs: Annotated[int, typer.Option("--num-epochs", help="Number of training epochs")] = 2,
    steps_per_epoch: Annotated[Optional[int], typer.Option("--steps-per-epoch", help="Steps per epoch")] = None,
    steps_per_epoch_val: Annotated[Optional[int], typer.Option("--steps-per-epoch-val", help="Validation steps per epoch")] = None,
    start_lr: Annotated[float, typer.Option("--start-lr", help="Initial learning rate")] = 5e-3,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 8,
    gpus: Annotated[str, typer.Option("--gpus", help="GPU IDs (comma-separated)")] = "",
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of data loader workers")] = 0,
    use_amp: Annotated[bool, typer.Option("--use-amp", help="Use automatic mixed precision")] = False,
    lr_scheduler: Annotated[str, typer.Option("--lr-scheduler", help="Learning rate scheduler")] = "none",
    tensorboard: Annotated[Optional[str], typer.Option("--tensorboard", help="TensorBoard log directory")] = None,
    backend: Annotated[str, typer.Option("--backend", help="Distributed backend")] = "none",
    model_prefix: Annotated[str, typer.Option("-m", "--model-prefix", help="Model checkpoint prefix")] = "checkpoints/network",
    task_type: Annotated[str, typer.Option("--task-type", help="Task type")] = "cls",
    experiment_name: Annotated[Optional[str], typer.Option("--experiment-name", help="MLflow experiment name")] = None,
) -> None:
    """Train a model."""
    # Convert network_option list to list of tuples
    network_options = []
    for i in range(0, len(network_option), 2):
        if i + 1 < len(network_option):
            network_options.append((network_option[i], network_option[i + 1]))

    # Create args object
    class Args:
        pass
    args = Args()
    args.data_config = data_config
    args.network_config = network_config
    args.train_files = parse_filespecs(data_train)
    args.val_files = parse_filespecs(data_val) if data_val else None
    args.extra_selection = extra_selection
    args.data_fraction = data_fraction
    args.file_fraction = file_fraction
    args.fetch_by_files = fetch_by_files
    args.fetch_step = fetch_step
    args.in_memory = in_memory
    args.train_val_split = train_val_split
    args.no_remake_weights = no_remake_weights
    args.log = log
    args.network_option = network_options
    args.num_epochs = num_epochs
    args.steps_per_epoch = steps_per_epoch
    args.steps_per_epoch_val = steps_per_epoch_val
    args.start_lr = start_lr
    args.batch_size = batch_size
    args.gpus = gpus
    args.num_workers = num_workers
    args.use_amp = use_amp
    args.lr_scheduler = lr_scheduler
    args.tensorboard = tensorboard
    args.backend = backend
    args.model_prefix = model_prefix
    args.task_type = task_type
    args.experiment_name = experiment_name

    run_train(args)


@app.command()
def predict(
    data_config: Annotated[str, typer.Option("-c", "--data-config", help="Path to data configuration YAML")],
    network_config: Annotated[str, typer.Option("-n", "--network-config", help="Path to network configuration Python file")],
    model_prefix: Annotated[str, typer.Option("-m", "--model-prefix", help="Model checkpoint path")],
    data_test: Annotated[List[str], typer.Option("-t", "--data-test", help="Test data files")] = [],
    data_fraction: Annotated[float, typer.Option("--data-fraction", help="Fraction of data to use")] = 1.0,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 8,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of data loader workers")] = 0,
    gpus: Annotated[str, typer.Option("--gpus", help="GPU IDs (comma-separated)")] = "",
    predict_output: Annotated[Optional[str], typer.Option("--predict-output", help="Output file path")] = None,
) -> None:
    """Run inference on a model."""
    class Args:
        pass
    args = Args()
    args.data_config = data_config
    args.network_config = network_config
    args.model_prefix = model_prefix
    args.test_files = parse_filespecs(data_test)
    args.data_fraction = data_fraction
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.gpus = gpus
    args.predict_output = predict_output

    run_predict(args)


@app.command()
def export(
    data_config: Annotated[str, typer.Option("-c", "--data-config", help="Path to data configuration YAML")],
    network_config: Annotated[str, typer.Option("-n", "--network-config", help="Path to network configuration Python file")],
    model_prefix: Annotated[str, typer.Option("-m", "--model-prefix", help="Model checkpoint path")],
    export_onnx: Annotated[str, typer.Option("--export-onnx", help="Output ONNX file path")],
    onnx_opset: Annotated[int, typer.Option("--onnx-opset", help="ONNX opset version")] = 15,
) -> None:
    """Export a model to ONNX format."""
    class Args:
        pass
    args = Args()
    args.data_config = data_config
    args.network_config = network_config
    args.model_prefix = model_prefix
    args.export_onnx = export_onnx
    args.onnx_opset = onnx_opset

    run_export(args)


@app.command()
def llm_finetune(
    model_name: Annotated[str, typer.Option("--model-name", help="Base model name")],
    lora_r: Annotated[int, typer.Option("--lora-r", help="LoRA rank")] = 8,
    lora_alpha: Annotated[float, typer.Option("--lora-alpha", help="LoRA alpha")] = 16,
    lora_dropout: Annotated[float, typer.Option("--lora-dropout", help="LoRA dropout")] = 0.1,
    target_modules: Annotated[List[str], typer.Option("--target-modules", help="Target modules for LoRA")] = ["q_proj", "v_proj"],
) -> None:
    """Fine-tune an LLM using LoRA."""
    class Args:
        pass
    args = Args()
    args.model_name = model_name
    args.lora_r = lora_r
    args.lora_alpha = lora_alpha
    args.lora_dropout = lora_dropout
    args.target_modules = target_modules

    run_lora(args)


@app.command()
def data_inspect(
    data_config: Annotated[str, typer.Option("-c", "--data-config", help="Path to data configuration YAML")],
    data_files: Annotated[List[str], typer.Option("-i", "--data-files", help="Data files to inspect")] = [],
) -> None:
    """Inspect data configuration and files."""
    from ..tasks.inspect import run as run_inspect

    class Args:
        pass
    args = Args()
    args.data_config = data_config
    args.files = parse_filespecs(data_files)

    run_inspect(args)


@app.command()
def submit(
    script: Annotated[str, typer.Option("--script", help="Script file path")],
    cmdline: Annotated[str, typer.Option("--cmdline", help="Command line to execute")],
    system: Annotated[str, typer.Option("--system", help="System type (local/slurm)")] = "local",
) -> None:
    """Generate and submit job scripts."""
    from ..runner.submit import run as run_submit

    class Args:
        pass
    args = Args()
    args.script = script
    args.cmdline = cmdline
    args.system = system

    run_submit(args)


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
