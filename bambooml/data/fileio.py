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
