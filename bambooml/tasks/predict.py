import os
import ast
import torch
import numpy as np
from torch.utils.data import DataLoader
from ..data.dataset import SimpleIterDataset

def import_module(module_path, name='_module'):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run(args):
    file_dict = args.test_files
    data = SimpleIterDataset(file_dict, args.data_config, for_training=False, load_range_and_fraction=((0, 1), args.data_fraction), fetch_by_files=True, fetch_step=1, name='test')
    loader = DataLoader(data, num_workers=min(args.num_workers, len(sum(file_dict.values(), []))), batch_size=args.batch_size, drop_last=False, pin_memory=True)
    dev = torch.device('cpu')
    if args.gpus:
        dev = torch.device('cuda:0') if torch.cuda.is_available() else dev
    model_module = import_module(args.network_config)
    model, model_info = model_module.get_model(data.config)
    model.load_state_dict(torch.load(args.model_prefix if args.model_prefix.endswith('.pt') else args.model_prefix + '_best_epoch_state.pt', map_location=dev))
    model = model.to(dev)
    scores = []
    labels = []
    observers = {}
    with torch.no_grad():
        for X, y, Z in loader:
            inputs = [torch.tensor(X[k], dtype=torch.float32, device=dev) for k in X.keys()]
            out = model(*inputs)
            scores.append(out.cpu().numpy())
            labels.append(next(iter(y.values())))
            for k, v in Z.items():
                observers.setdefault(k, []).append(v)
    scores = np.concatenate(scores)
    labels = {next(iter(data.config.label_names)): np.concatenate(labels)}
    for k, arrs in observers.items():
        observers[k] = np.concatenate(arrs)
    if args.predict_output:
        import pyarrow as pa
        import pyarrow.parquet as pq
        t = pa.table({'scores': [scores], **labels, **observers})
        os.makedirs(os.path.dirname(args.predict_output), exist_ok=True)
        pq.write_table(t, args.predict_output)
