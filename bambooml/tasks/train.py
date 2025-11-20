import os
import ast
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..monitor.tensorboard import get_writer, log_scalar
from ..core.logging import get_logger
from ..data.dataset import SimpleIterDataset

def import_module(module_path, name='_module'):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def model_setup(args, data_config, device='cpu'):
    network_module = import_module(args.network_config, name='_network_module')
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    model, model_info = network_module.get_model(data_config, **network_options)
    try:
        loss_func = network_module.get_loss(data_config, **network_options)
    except AttributeError:
        if args.task_type == 'reg':
            loss_func = torch.nn.MSELoss()
        elif args.task_type == 'cls':
            loss_func = torch.nn.CrossEntropyLoss()
        else:
            loss_func = None
    return model, model_info, loss_func

def _prepare_inputs(X, device):
    def _to_tensor(x, dtype, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=device)
    return [_to_tensor(X[k], torch.float32, device) for k in X.keys()]

def _prepare_labels(y, device, task_type):
    if task_type == 'multitask':
        result = {}
        for k, v in y.items():
            if isinstance(v, torch.Tensor):
                arr = v.to(device=device)
            else:
                arr = torch.as_tensor(v, device=device)
            if arr.dtype in (torch.int32, torch.int64):
                result[k] = arr.long()
            else:
                result[k] = arr.float()
        return result
    else:
        v = list(y.values())[0]
        if task_type == 'reg':
            return torch.as_tensor(v, dtype=torch.float32, device=device) if not isinstance(v, torch.Tensor) else v.to(device=device, dtype=torch.float32)
        return torch.as_tensor(v, dtype=torch.long, device=device) if not isinstance(v, torch.Tensor) else v.to(device=device, dtype=torch.long)

def _compute_loss(out, labels, loss_func, task_type):
    if task_type != 'multitask':
        return loss_func(out, labels)
    if loss_func is not None:
        return loss_func(out, labels)
    total = 0.0
    if isinstance(out, dict):
        for k, tgt in labels.items():
            pred = out[k]
            if tgt.dtype == torch.long and pred.dim() > 1:
                total = total + torch.nn.functional.cross_entropy(pred, tgt)
            else:
                total = total + torch.nn.functional.mse_loss(pred.squeeze(), tgt.float())
        return total
    if isinstance(out, (tuple, list)):
        keys = list(labels.keys())
        for idx, pred in enumerate(out):
            tgt = labels[keys[idx]]
            if tgt.dtype == torch.long and pred.dim() > 1:
                total = total + torch.nn.functional.cross_entropy(pred, tgt)
            else:
                total = total + torch.nn.functional.mse_loss(pred.squeeze(), tgt.float())
        return total
    if isinstance(labels, dict):
        stacked = torch.stack([labels[k].float() for k in labels.keys()], dim=1)
        return torch.nn.functional.mse_loss(out.squeeze(), stacked)
    return torch.nn.functional.mse_loss(out.squeeze(), labels.float())

def train_loop(model, loss_func, opt, loader, device, steps_per_epoch=None, amp=False, scaler=None, task_type='cls'):
    model.train()
    if steps_per_epoch is None:
        for X, y, _ in loader:
            inputs = _prepare_inputs(X, device)
            labels = _prepare_labels(y, device, task_type)
            opt.zero_grad()
            if amp:
                with torch.cuda.amp.autocast():
                    out = model(*inputs)
                    loss = _compute_loss(out, labels, loss_func, task_type)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(*inputs)
                loss = _compute_loss(out, labels, loss_func, task_type)
                loss.backward()
                opt.step()
    else:
        it = iter(loader)
        for _ in range(steps_per_epoch):
            X, y, _ = next(it)
            inputs = _prepare_inputs(X, device)
            labels = _prepare_labels(y, device, task_type)
            opt.zero_grad()
            if amp:
                with torch.cuda.amp.autocast():
                    out = model(*inputs)
                    loss = _compute_loss(out, labels, loss_func, task_type)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(*inputs)
                loss = _compute_loss(out, labels, loss_func, task_type)
                loss.backward()
                opt.step()

def evaluate_loop(model, loss_func, loader, device, steps_per_epoch=None, task_type='cls'):
    model.eval()
    losses = []
    with torch.no_grad():
        if steps_per_epoch is None:
            for X, y, _ in loader:
                inputs = _prepare_inputs(X, device)
                labels = _prepare_labels(y, device, task_type)
                out = model(*inputs)
                loss = _compute_loss(out, labels, loss_func, task_type)
                losses.append(float(loss.item()))
        else:
            it = iter(loader)
            for _ in range(steps_per_epoch):
                X, y, _ = next(it)
                inputs = _prepare_inputs(X, device)
                labels = _prepare_labels(y, device, task_type)
                out = model(*inputs)
                loss = _compute_loss(out, labels, loss_func, task_type)
                losses.append(float(loss.item()))
    return float(np.mean(losses))

def run(args):
    logger = get_logger('bambooml', stdout=True, filename=args.log if args.log else None)
    train_file_dict = args.train_files
    val_file_dict = args.val_files or train_file_dict
    train_range = (0, args.train_val_split)
    val_range = (args.train_val_split, 1)
    train_data = SimpleIterDataset(train_file_dict, args.data_config, for_training=True, extra_selection=args.extra_selection, remake_weights=not args.no_remake_weights, load_range_and_fraction=(train_range, args.data_fraction), file_fraction=args.file_fraction, fetch_by_files=args.fetch_by_files, fetch_step=args.fetch_step, infinity_mode=args.steps_per_epoch is not None, in_memory=args.in_memory, name='train')
    val_data = SimpleIterDataset(val_file_dict, args.data_config, for_training=True, extra_selection=args.extra_selection, load_range_and_fraction=(val_range, args.data_fraction), file_fraction=args.file_fraction, fetch_by_files=args.fetch_by_files, fetch_step=args.fetch_step, infinity_mode=args.steps_per_epoch_val is not None, in_memory=args.in_memory, name='val')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=min(args.num_workers, int(len(sum(train_file_dict.values(), [])) * args.file_fraction)))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=min(args.num_workers, int(len(sum(val_file_dict.values(), [])) * args.file_fraction)))
    data_config = train_data.config
    dev = torch.device('cpu')
    if args.gpus:
        dev = torch.device('cuda:0') if torch.cuda.is_available() else dev
    model, model_info, loss_func = model_setup(args, data_config, device=dev)
    if args.gpus and ',' in args.gpus:
        model = torch.nn.DataParallel(model)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.start_lr)
    if args.lr_scheduler == 'flat+decay':
        num_decay_epochs = max(1, int(args.num_epochs * 0.3))
        milestones = list(range(args.num_epochs - num_decay_epochs, args.num_epochs))
        gamma = 0.01 ** (1. / num_decay_epochs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
    else:
        scheduler = None
    writer = get_writer(args.tensorboard) if args.tensorboard else None
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and (dev.type == 'cuda') else None
    import datetime as _dt
    import socket as _sk
    if args.model_prefix:
        base_dir = os.path.dirname(args.model_prefix) if os.path.dirname(args.model_prefix) else 'checkpoints'
        base_name = os.path.basename(args.model_prefix)
        exp_name = os.path.basename(args.tensorboard) if args.tensorboard else 'default'
        run_stamp = _dt.datetime.now().strftime('%b%d_%H-%M-%S') + '_' + _sk.gethostname() + 'runs'
        ckpt_dir = os.path.join(base_dir, run_stamp, exp_name)
        model_prefix_effective = os.path.join(ckpt_dir, base_name)
    else:
        model_prefix_effective = None
    best_valid = float('inf')
    for epoch in range(args.num_epochs):
        train_loop(model, loss_func, opt, train_loader, dev, steps_per_epoch=args.steps_per_epoch, amp=args.use_amp, scaler=scaler, task_type=args.task_type)
        valid = evaluate_loop(model, loss_func, val_loader, dev, steps_per_epoch=args.steps_per_epoch_val, task_type=args.task_type)
        logger.info(f'Epoch {epoch} valid {valid}')
        log_scalar(writer, 'valid/loss', valid, epoch)
        if scheduler is not None:
            scheduler.step()
        if model_prefix_effective:
            state_dict = model.state_dict()
            os.makedirs(os.path.dirname(model_prefix_effective), exist_ok=True)
            torch.save(state_dict, model_prefix_effective + f'_epoch-{epoch}_state.pt')
        if valid < best_valid:
            best_valid = valid
            if model_prefix_effective:
                import shutil
                shutil.copy2(model_prefix_effective + f'_epoch-{epoch}_state.pt', model_prefix_effective + '_best_epoch_state.pt')
