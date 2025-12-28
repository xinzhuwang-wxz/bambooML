import math
from .tools import _concat

def _read_hdf5(filepath, branches, load_range=None):
    import tables
    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k: getattr(f.root, k)[:] for k in branches}
    if load_range is None:
        load_range = (0, 1)
    start = math.trunc(load_range[0] * len(outputs[branches[0]]))
    stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
    for k, v in outputs.items():
        outputs[k] = v[start:stop]
    return outputs

def _read_root(filepath, branches, load_range=None, treename=None, branch_magic=None):
    
    '''
    _read_root 用于从 ROOT 文件中一次性加载多个 branch，返回一个
    {branch_name: numpy.ndarray} 的字典。常用示例：

    1）标量 double branch：
        >>> arrs = _read_root('file.root', ['jet_pt', 'jet_eta'])
        >>> arrs['jet_pt']   # dtype=float64，形状为 (n_entries,)
        array([42.1, 38.5, ...])

    2）矢量类型（std::vector<double>）：
        ROOT 中的 vector<double> 会被 uproot 解出为 object dtype 的一维数组，
        每个元素本身是一个 numpy.ndarray。例如：
        >>> arrs = _read_root('file.root', ['jet_constituent_pt'])
        >>> arrs['jet_constituent_pt'][0]  # 第 0 个事件的全部 constituent pt
        array([1.2, 0.8, 0.5])

       如需将变长 vector 展平成固定长度，可在上层逻辑中自行 pad/truncate。

    3）branch_magic 的用途：
        若 ROOT 文件中的真实 branch 名与代码中想要的别名不一致，可以传入
        branch_magic 做字符串替换。例如：
        >>> branch_magic = {'jet': 'Jet.', 'pt': 'PT'}
        >>> _read_root('file.root', ['jet_pt'], branch_magic=branch_magic)
        上述调用会把 'jet_pt' 转换为 'Jet.PT' 去读取，但最终字典的 key 仍是 'jet_pt'。

    load_range=(start_frac, end_frac) 用于按比例读取部分 entries，treename 可显式指定 TTree。
    读取失败会抛出 RuntimeError。
    '''

    import uproot
    with uproot.open(filepath) as f:
        if treename is None:
            treenames = set([k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError('Need treename for file %s' % filepath)
        tree = f[treename]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.num_entries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        else:
            start, stop = None, None
        if branch_magic is not None:
            branch_dict = {}
            for name in branches:
                decoded_name = name
                for src, tgt in branch_magic.items():
                    if src in decoded_name:
                        decoded_name = decoded_name.replace(src, tgt)
                branch_dict[name] = decoded_name
            outputs = tree.arrays(filter_name=list(branch_dict.values()), entry_start=start, entry_stop=stop)
            for name, decoded_name in branch_dict.items():
                if name != decoded_name:
                    outputs[name] = outputs[decoded_name]
        else:
            outputs = tree.arrays(filter_name=branches, entry_start=start, entry_stop=stop)
    return {k: outputs[k] for k in outputs.fields}

def _read_parquet(filepath, branches, load_range=None):
    import pyarrow.parquet as pq
    table = pq.read_table(filepath, columns=branches)
    outputs = {name: table[name].to_numpy() for name in branches}
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs[branches[0]]))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
        for k, v in outputs.items():
            outputs[k] = v[start:stop]
    return outputs

def _read_csv(filepath, branches, load_range=None):
    import pandas as pd
    df = pd.read_csv(filepath, usecols=branches)
    outputs = {k: df[k].to_numpy() for k in branches}
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs[branches[0]]))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
        for k, v in outputs.items():
            outputs[k] = v[start:stop]
    return outputs

def _read_files(filelist, branches, load_range=None, show_progressbar=False, file_magic=None, **kwargs):
    import os
    branches = list(branches)
    table = []
    flist = filelist
    for filepath in flist:
        ext = os.path.splitext(filepath)[1]
        if ext not in ('.h5', '.root', '.parquet', '.csv'):
            raise RuntimeError('Unsupported file type %s' % ext)
        try:
            if ext == '.h5':
                a = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == '.root':
                a = _read_root(filepath, branches, load_range=load_range, treename=kwargs.get('treename', None), branch_magic=kwargs.get('branch_magic', None))
            elif ext == '.parquet':
                a = _read_parquet(filepath, branches, load_range=load_range)
            elif ext == '.csv':
                a = _read_csv(filepath, branches, load_range=load_range)
        except Exception:
            a = None
        if a is not None:
            if file_magic is not None:
                import re
                for var, value_dict in file_magic.items():
                    a[var] = 0
                    for fn_pattern, value in value_dict.items():
                        if re.search(fn_pattern, filepath):
                            a[var] = value
                            break
            table.append(a)
    arrays = {}
    if len(table) == 0:
        raise RuntimeError('Zero entries loaded')
    for k in branches:
        arrays[k] = _concat([t[k] for t in table])
    for k in [kk for kk in table[0].keys() if kk not in branches]:
        arrays[k] = _concat([t[k] for t in table if k in t])
    return arrays
