import os
import torch
from ..data.config import DataConfig

def run(args):
    assert args.export_onnx.endswith('.onnx')
    model_path = args.model_prefix
    data_config = DataConfig.load(args.data_config, load_observers=False, load_reweight_info=False)
    mod = __import__('importlib').import_module('importlib.util')
    spec = mod.spec_from_file_location('_network_module', args.network_config)
    net = mod.module_from_spec(spec)
    spec.loader.exec_module(net)
    model, model_info = net.get_model(data_config, for_inference=True)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.cpu()
    model.eval()
    os.makedirs(os.path.dirname(args.export_onnx), exist_ok=True)
    inputs = tuple(torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])
    torch.onnx.export(model, inputs, args.export_onnx, input_names=model_info['input_names'], output_names=model_info['output_names'], dynamic_axes=model_info.get('dynamic_axes', None), opset_version=args.onnx_opset)
    preprocessing_json = os.path.join(os.path.dirname(args.export_onnx), 'preprocess.json')
    data_config.export_json(preprocessing_json)
