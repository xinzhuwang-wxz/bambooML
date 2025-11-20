import copy
import numpy as np
from .tools import _get_variable_names, _eval_expr
from .fileio import _read_files

def _apply_selection(table, selection, funcs=None):
    if selection is None:
        return table
    if funcs:
        new_vars = {k: funcs[k] for k in _get_variable_names(selection) if k not in table and k in funcs}
        _build_new_variables(table, new_vars)
    selected = np.asarray(_eval_expr(selection, table)).astype('bool')
    return {k: v[selected] for k, v in table.items()}

def _build_new_variables(table, funcs):
    if funcs is None:
        return table
    for k, expr in funcs.items():
        if k in table:
            continue
        table[k] = _eval_expr(expr, table)
    return table

def _build_weights(table, data_config, reweight_hists=None):
    if data_config.weight_name is None:
        raise RuntimeError('weight_name is None')
    if data_config.use_precomputed_weights:
        return table[data_config.weight_name]
    x_var, y_var = data_config.reweight_branches
    x_bins, y_bins = data_config.reweight_bins
    rwgt_sel = None
    if data_config.reweight_discard_under_overflow:
        rwgt_sel = (table[x_var] >= min(x_bins)) & (table[x_var] <= max(x_bins)) & (table[y_var] >= min(y_bins)) & (table[y_var] <= max(y_bins))
    wgt = np.zeros(len(table[x_var]), dtype='float32')
    sum_evts = 0
    if reweight_hists is None:
        reweight_hists = data_config.reweight_hists
    for label, hist in reweight_hists.items():
        pos = table[label] == 1
        if rwgt_sel is not None:
            pos = (pos & rwgt_sel)
        rwgt_x_vals = table[x_var][pos]
        rwgt_y_vals = table[y_var][pos]
        x_indices = np.clip(np.digitize(rwgt_x_vals, x_bins) - 1, a_min=0, a_max=len(x_bins) - 2)
        y_indices = np.clip(np.digitize(rwgt_y_vals, y_bins) - 1, a_min=0, a_max=len(y_bins) - 2)
        wgt[pos] = hist[x_indices, y_indices]
        sum_evts += int(np.sum(pos))
    if data_config.reweight_basewgt:
        wgt *= table[data_config.basewgt_name]
    return wgt

class AutoStandardizer:
    def __init__(self, filelist, data_config):
        if isinstance(filelist, dict):
            filelist = sum(filelist.values(), [])
        self._filelist = filelist if isinstance(filelist, (list, tuple)) else []
        self._data_config = data_config.copy()
        self.load_range = (0, data_config.preprocess.get('data_fraction', 0.1))

    def read_file(self, filelist):
        keep_branches = set()
        aux_branches = set()
        load_branches = set()
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] == 'auto':
                keep_branches.add(k)
                load_branches.add(k)
        if self._data_config.selection:
            load_branches.update(_get_variable_names(self._data_config.selection))
        func_vars = set(self._data_config.var_funcs.keys())
        while load_branches & func_vars:
            for k in (load_branches & func_vars):
                aux_branches.add(k)
                load_branches.remove(k)
                load_branches.update(_get_variable_names(self._data_config.var_funcs[k]))
        table = _read_files(filelist, load_branches, self.load_range, treename=self._data_config.treename, branch_magic=self._data_config.branch_magic, file_magic=self._data_config.file_magic)
        table = _apply_selection(table, self._data_config.selection, funcs=self._data_config.var_funcs)
        table = _build_new_variables(table, {k: v for k, v in self._data_config.var_funcs.items() if k in aux_branches})
        table = {k: table[k] for k in keep_branches}
        return table

    def make_preprocess_params(self, table):
        preprocess_params = copy.deepcopy(self._data_config.preprocess_params)
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] == 'auto':
                a = table[k]
                a = a[np.isfinite(a)]
                if a.size == 0:
                    center = 0.0
                    scale = 1.0
                else:
                    low, center, high = np.percentile(a, [16, 50, 84])
                    scale = max(high - center, center - low)
                    scale = 1 if scale == 0 else 1.0 / scale
                params['center'] = float(center)
                params['scale'] = float(scale)
                preprocess_params[k] = params
        return preprocess_params

    def produce(self, output=None):
        table = self.read_file(self._filelist)
        preprocess_params = self.make_preprocess_params(table)
        self._data_config.preprocess_params = preprocess_params
        self._data_config.options['preprocess']['params'] = preprocess_params
        if output:
            self._data_config.dump(output)
        return self._data_config

class WeightMaker:
    def __init__(self, filelist, data_config):
        if isinstance(filelist, dict):
            filelist = sum(filelist.values(), [])
        self._filelist = filelist if isinstance(filelist, (list, tuple)) else []
        self._data_config = data_config.copy()

    def read_file(self, filelist):
        keep_branches = set(self._data_config.reweight_branches + self._data_config.reweight_classes)
        if self._data_config.reweight_basewgt:
            keep_branches.add(self._data_config.basewgt_name)
        aux_branches = set()
        load_branches = keep_branches.copy()
        if self._data_config.selection:
            load_branches.update(_get_variable_names(self._data_config.selection))
        func_vars = set(self._data_config.var_funcs.keys())
        while load_branches & func_vars:
            for k in (load_branches & func_vars):
                aux_branches.add(k)
                load_branches.remove(k)
                load_branches.update(_get_variable_names(self._data_config.var_funcs[k]))
        table = _read_files(filelist, load_branches, treename=self._data_config.treename, branch_magic=self._data_config.branch_magic, file_magic=self._data_config.file_magic)
        table = _apply_selection(table, self._data_config.selection, funcs=self._data_config.var_funcs)
        table = _build_new_variables(table, {k: v for k, v in self._data_config.var_funcs.items() if k in aux_branches})
        return {k: table[k] for k in keep_branches}

    def make_weights(self, table):
        x_var, y_var = self._data_config.reweight_branches
        x_bins, y_bins = self._data_config.reweight_bins
        if not self._data_config.reweight_discard_under_overflow:
            x_min, x_max = min(x_bins), max(x_bins)
            y_min, y_max = min(y_bins), max(y_bins)
            table[x_var] = np.clip(table[x_var], x_min, x_max)
            table[y_var] = np.clip(table[y_var], y_min, y_max)
        sum_evts = 0
        raw_hists = {}
        class_events = {}
        result = {}
        for label in self._data_config.reweight_classes:
            pos = table[label] == 1
            x = table[x_var][pos]
            y = table[y_var][pos]
            hist, _, _ = np.histogram2d(x, y, bins=self._data_config.reweight_bins)
            sum_evts += int(hist.sum())
            if self._data_config.reweight_basewgt:
                w = table[self._data_config.basewgt_name][pos]
                hist, _, _ = np.histogram2d(x, y, weights=w, bins=self._data_config.reweight_bins)
            raw_hists[label] = hist.astype('float32')
            result[label] = hist.astype('float32')
        if self._data_config.reweight_method == 'flat':
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                hist = result[label]
                threshold_ = np.median(hist[hist > 0]) * 0.01
                nonzero_vals = hist[hist > threshold_]
                ref_val = np.percentile(nonzero_vals, self._data_config.reweight_threshold) if nonzero_vals.size else 1.0
                wgt = np.clip(np.nan_to_num(ref_val / hist, posinf=0), 0, 1)
                result[label] = wgt
                class_events[label] = float(np.sum(raw_hists[label] * wgt)) / max(1e-9, classwgt)
        elif self._data_config.reweight_method == 'ref':
            hist_ref = raw_hists[self._data_config.reweight_classes[0]]
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                ratio = np.nan_to_num(hist_ref / result[label], posinf=0)
                upper = np.percentile(ratio[ratio > 0], 100 - self._data_config.reweight_threshold) if np.any(ratio > 0) else 1.0
                wgt = np.clip(ratio / upper, 0, 1)
                result[label] = wgt
                class_events[label] = float(np.sum(raw_hists[label] * wgt)) / max(1e-9, classwgt)
        max_weight = 0.9
        min_nevt = min(class_events.values()) * max_weight if len(class_events) else 1.0
        for label in self._data_config.reweight_classes:
            class_wgt = float(min_nevt) / max(1e-9, class_events[label])
            result[label] *= class_wgt
        if self._data_config.reweight_basewgt:
            wgts = _build_weights(table, self._data_config, reweight_hists=result)
            wgt_ref = np.percentile(wgts, 100 - self._data_config.reweight_threshold) if wgts.size else 1.0
            for label in self._data_config.reweight_classes:
                result[label] = result[label] / max(1e-9, wgt_ref)
        return result

    def produce(self, output=None):
        table = self.read_file(self._filelist)
        wgts = self.make_weights(table)
        self._data_config.reweight_hists = wgts
        self._data_config.options['weights']['reweight_hists'] = {k: v.tolist() for k, v in wgts.items()}
        if output:
            self._data_config.dump(output)
        return self._data_config
