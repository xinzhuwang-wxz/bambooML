import numpy as np
from ..data.config import DataConfig
from ..data.fileio import _read_files
from ..data.preprocess import _apply_selection, _build_new_variables

'''
inspect 是一个模块，用于检查数据集。比如检查数据集的分布、数据集的形状、数据集的类型等。
'''
def run(args):
    dc = DataConfig.load(args.data_config)
    file_dict = args.files
    filelist = sum(file_dict.values(), [])
    branches = list(dc.train_load_branches or [])
    if not branches:
        for names in dc.input_dicts.values():
            branches += list(names)
    table = _read_files(filelist, branches, load_range=(0, 1), treename=dc.treename, branch_magic=dc.branch_magic, file_magic=dc.file_magic)
    table = _apply_selection(table, dc.selection, funcs=dc.var_funcs)
    table = _build_new_variables(table, dc.var_funcs)
    stats = {}
    for k, v in table.items():
        arr = np.asarray(v)
        if arr.ndim == 1:
            p = np.percentile(arr, [0, 25, 50, 75, 100])
            stats[k] = {'count': int(arr.size), 'min': float(p[0]), 'p50': float(p[2]), 'max': float(p[4])}
        else:
            stats[k] = {'shape': list(arr.shape)}
    for k, s in stats.items():
        print(k, s)
