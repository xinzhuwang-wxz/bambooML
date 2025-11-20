import argparse
import glob
import os
from ..tasks.train import run as run_train
from ..tasks.predict import run as run_predict
from ..tasks.export import run as run_export
from ..llm.lora import run as run_lora

def parse_filespecs(specs):
    file_dict = {}
    for f in specs:
        if ':' in f:
            name, fp = f.split(':')
        else:
            name, fp = '_', f
        files = glob.glob(fp)
        file_dict[name] = sorted(files)
    return file_dict

def main():
    p = argparse.ArgumentParser(prog='bambooml')
    sub = p.add_subparsers(dest='cmd')
    pt = sub.add_parser('train')
    pt.add_argument('-c', '--data-config', type=str, required=True)
    pt.add_argument('--extra-selection', type=str, default=None)
    pt.add_argument('-i', '--data-train', nargs='*', default=[])
    pt.add_argument('-l', '--data-val', nargs='*', default=[])
    pt.add_argument('--data-fraction', type=float, default=1)
    pt.add_argument('--file-fraction', type=float, default=1)
    pt.add_argument('--fetch-by-files', action='store_true', default=False)
    pt.add_argument('--fetch-step', type=float, default=0.01)
    pt.add_argument('--in-memory', action='store_true', default=False)
    pt.add_argument('--train-val-split', type=float, default=0.8)
    pt.add_argument('--no-remake-weights', action='store_true', default=False)
    pt.add_argument('--log', type=str, default='')
    pt.add_argument('-n', '--network-config', type=str, required=True)
    pt.add_argument('-o', '--network-option', nargs=2, action='append', default=[])
    pt.add_argument('--num-epochs', type=int, default=2)
    pt.add_argument('--steps-per-epoch', type=int, default=None)
    pt.add_argument('--steps-per-epoch-val', type=int, default=None)
    pt.add_argument('--start-lr', type=float, default=5e-3)
    pt.add_argument('--batch-size', type=int, default=8)
    pt.add_argument('--gpus', type=str, default='')
    pt.add_argument('--num-workers', type=int, default=0)
    pt.add_argument('--use-amp', action='store_true', default=False)
    pt.add_argument('--lr-scheduler', type=str, choices=['none','flat+decay'], default='none')
    pt.add_argument('--tensorboard', type=str, default=None)
    pt.add_argument('--backend', type=str, choices=['none','gloo','nccl','mpi'], default='none')
    pt.add_argument('-m', '--model-prefix', type=str, default='checkpoints/network')
    pt.add_argument('--task-type', type=str, choices=['cls','reg','multitask'], default='cls')
    pp = sub.add_parser('predict')
    pp.add_argument('-c', '--data-config', type=str, required=True)
    pp.add_argument('-t', '--data-test', nargs='*', default=[])
    pp.add_argument('--data-fraction', type=float, default=1)
    pp.add_argument('--batch-size', type=int, default=8)
    pp.add_argument('--num-workers', type=int, default=0)
    pp.add_argument('--gpus', type=str, default='')
    pp.add_argument('-n', '--network-config', type=str, required=True)
    pp.add_argument('-m', '--model-prefix', type=str, required=True)
    pp.add_argument('--predict-output', type=str, required=False)
    pe = sub.add_parser('export')
    pe.add_argument('-c', '--data-config', type=str, required=True)
    pe.add_argument('-n', '--network-config', type=str, required=True)
    pe.add_argument('-m', '--model-prefix', type=str, required=True)
    pe.add_argument('--export-onnx', type=str, required=True)
    pe.add_argument('--onnx-opset', type=int, default=15)
    pl = sub.add_parser('llm-finetune')
    pl.add_argument('--model-name', type=str, required=True)
    pl.add_argument('--lora-r', type=int, default=8)
    pl.add_argument('--lora-alpha', type=float, default=16)
    pl.add_argument('--lora-dropout', type=float, default=0.1)
    pl.add_argument('--target-modules', nargs='*', default=['q_proj','v_proj'])
    pi = sub.add_parser('data-inspect')
    pi.add_argument('-c', '--data-config', type=str, required=True)
    pi.add_argument('-i', '--data-files', nargs='*', default=[])
    ps = sub.add_parser('submit')
    ps.add_argument('--system', type=str, choices=['local','slurm'], default='local')
    ps.add_argument('--script', type=str, required=True)
    ps.add_argument('--cmdline', type=str, required=True)
    args = p.parse_args()
    if args.cmd == 'train':
        args.train_files = parse_filespecs(args.data_train)
        args.val_files = parse_filespecs(args.data_val) if args.data_val else None
        return run_train(args)
    if args.cmd == 'predict':
        args.test_files = parse_filespecs(args.data_test)
        return run_predict(args)
    if args.cmd == 'export':
        return run_export(args)
    if args.cmd == 'llm-finetune':
        return run_lora(args)
    if args.cmd == 'data-inspect':
        from ..tasks.inspect import run as run_inspect
        args.files = parse_filespecs(args.data_files)
        return run_inspect(args)
    if args.cmd == 'submit':
        from ..runner.submit import run as run_submit
        return run_submit(args)
    p.print_help()

if __name__ == '__main__':
    main()
