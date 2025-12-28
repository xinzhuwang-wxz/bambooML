import copy
import numpy as np
import torch.utils.data
from functools import partial
from concurrent.futures.thread import ThreadPoolExecutor
from .dataio_adapter import stack_groups
from .fileio import _read_files
from .config import DataConfig, _md5
from .preprocess import _apply_selection, _build_new_variables, _build_weights

def _finalize_inputs(table, data_config):
    output = {}
    for k in data_config.z_variables:
        if k in data_config.observer_names:
            output[k] = table[k]
    for k in data_config.label_names:
        output[k] = table[k]
    for k, params in data_config.preprocess_params.items():
        if data_config._auto_standardization and params['center'] == 'auto':
            raise ValueError('No valid standardization params for %s' % k)
        if params['center'] is not None:
            table[k] = np.clip((table[k] - params['center']) * params['scale'], params['min'], params['max'])
        if params['length'] is not None:
            pad_fn = partial(stack_groups.pad_array, value=params['pad_value']) if params['pad_mode'] != 'wrap' else stack_groups.repeat_pad
            table[k] = pad_fn(table[k], params['length'])
        if np.any(np.isnan(table[k])):
            table[k] = np.nan_to_num(table[k])
    for k, names in data_config.input_dicts.items():
        if len(names) == 1 and data_config.preprocess_params[names[0]]['length'] is None:
            output['_' + k] = np.asarray(table[names[0]]).astype('float32')
        else:
            output['_' + k] = np.stack([np.asarray(table[n]).astype('float32') for n in names], axis=1)
    for k in data_config.z_variables:
        if k in data_config.monitor_variables:
            output[k] = table[k]
    return output

def _get_reweight_indices(weights, up_sample=True, max_resample=10, weight_scale=1):
    all_indices = np.arange(len(weights))
    randwgt = np.random.uniform(low=0, high=weight_scale, size=len(weights))
    keep_flags = randwgt < weights
    if not up_sample:
        keep_indices = all_indices[keep_flags]
    else:
        n_repeats = len(weights) // max(1, int(keep_flags.sum()))
        if n_repeats > max_resample:
            n_repeats = max_resample
        all_indices = np.repeat(np.arange(len(weights)), n_repeats)
        randwgt = np.random.uniform(low=0, high=weight_scale, size=len(weights) * n_repeats)
        keep_indices = all_indices[randwgt < np.repeat(weights, n_repeats)]
    return copy.deepcopy(keep_indices)

def _check_labels(table):
    if np.all(table['_labelcheck_'] == 1):
        return
    else:
        if np.any(table['_labelcheck_'] == 0):
            raise RuntimeError('Inconsistent label definition: unassigned classes')
        if np.any(table['_labelcheck_'] > 1):
            raise RuntimeError('Inconsistent label definition: multiple classes')

def _preprocess(table, data_config, options):
    table = _apply_selection(table, data_config.selection if options['training'] else data_config.test_time_selection, funcs=data_config.var_funcs)
    if len(table[list(table.keys())[0]]) == 0:
        return []
    aux_branches = data_config.train_aux_branches if options['training'] else data_config.test_aux_branches
    table = _build_new_variables(table, {k: v for k, v in data_config.var_funcs.items() if k in aux_branches})
    if data_config.label_type == 'simple' and options['training']:
        _check_labels(table)
    if options['reweight'] and data_config.weight_name is not None:
        wgts = _build_weights(table, data_config)
        indices = _get_reweight_indices(wgts, up_sample=options['up_sample'], weight_scale=options['weight_scale'], max_resample=options['max_resample'])
    else:
        indices = np.arange(len(table[data_config.label_names[0]]))
    if options['shuffle']:
        np.random.shuffle(indices)
    table = _finalize_inputs(table, data_config)
    return table, indices

def _load_next(data_config, filelist, load_range, options):
    load_branches = data_config.train_load_branches if options['training'] else data_config.test_load_branches
    table = _read_files(filelist, load_branches, load_range, treename=data_config.treename, branch_magic=data_config.branch_magic, file_magic=data_config.file_magic)
    table, indices = _preprocess(table, data_config, options)
    return table, indices

class _SimpleIter(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.executor = ThreadPoolExecutor(max_workers=1) if self._async_load else None
        self.prefetch = None
        self.table = None
        self.indices = []
        self.cursor = 0
        self._seed = None
        worker_info = torch.utils.data.get_worker_info()
        file_dict = copy.deepcopy(self._init_file_dict)
        if worker_info is not None:
            self._name += '_worker%d' % worker_info.id
            self._seed = worker_info.seed & 0xFFFFFFFF
            np.random.seed(self._seed)
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[worker_info.id::worker_info.num_workers]
                assert len(new_files) > 0
                new_file_dict[name] = new_files
            file_dict = new_file_dict
        self.worker_file_dict = file_dict
        self.worker_filelist = sum(file_dict.values(), [])
        self.worker_info = worker_info
        self.restart()

    def restart(self):
        filelist = copy.deepcopy(self.worker_filelist)
        if self._sampler_options['shuffle']:
            np.random.shuffle(filelist)
        if self._file_fraction < 1:
            num_files = int(len(filelist) * self._file_fraction)
            filelist = filelist[:num_files]
        self.filelist = filelist
        if self._init_load_range_and_fraction is None:
            self.load_range = (0, 1)
        else:
            (start_pos, end_pos), load_frac = self._init_load_range_and_fraction
            interval = (end_pos - start_pos) * load_frac
            if self._sampler_options['shuffle']:
                offset = np.random.uniform(start_pos, end_pos - interval)
                self.load_range = (offset, offset + interval)
            else:
                self.load_range = (start_pos, start_pos + interval)
        self.ipos = 0 if self._fetch_by_files else self.load_range[0]
        self._try_get_next(init=True)

    def __next__(self):
        if len(self.filelist) == 0:
            raise StopIteration
        try:
            i = self.indices[self.cursor]
        except IndexError:
            while True:
                if self._in_memory and len(self.indices) > 0:
                    if self._sampler_options['shuffle']:
                        np.random.shuffle(self.indices)
                    break
                if self.prefetch is None:
                    self.table = None
                    if self._async_load:
                        self.executor.shutdown(wait=False)
                    raise StopIteration
                if self._async_load:
                    self.table, self.indices = self.prefetch.result()
                else:
                    self.table, self.indices = self.prefetch
                self._try_get_next()
                if len(self.indices) > 0:
                    break
            self.cursor = 0
            i = self.indices[self.cursor]
        self.cursor += 1
        return self.get_data(i)

    def _try_get_next(self, init=False):
        end_of_list = self.ipos >= len(self.filelist) if self._fetch_by_files else self.ipos >= self.load_range[1]
        if end_of_list:
            if init:
                raise RuntimeError('Nothing to load')
            if self._infinity_mode and not self._in_memory:
                self.restart()
                return
            else:
                self.prefetch = None
                return
        if self._fetch_by_files:
            filelist = self.filelist[int(self.ipos): int(self.ipos + self._fetch_step)]
            load_range = self.load_range
        else:
            filelist = self.filelist
            load_range = (self.ipos, min(self.ipos + self._fetch_step, self.load_range[1]))
        if self._async_load:
            self.prefetch = self.executor.submit(_load_next, self._data_config, filelist, load_range, self._sampler_options)
        else:
            self.prefetch = _load_next(self._data_config, filelist, load_range, self._sampler_options)
        self.ipos += self._fetch_step

    def get_data(self, i):
        X = {k: copy.deepcopy(self.table['_' + k][i]) for k in self._data_config.input_names}
        y = {k: copy.deepcopy(self.table[k][i]) for k in self._data_config.label_names}
        Z = {k: copy.deepcopy(self.table[k][i]) for k in self._data_config.z_variables}
        return X, y, Z

class SimpleIterDataset(torch.utils.data.IterableDataset):
    '''
    SimpleIterDataset 是一个类，用于创建一个简单的迭代器数据集。

    '''
    def __init__(self, file_dict, data_config_file, for_training=True, load_range_and_fraction=None, extra_selection=None, fetch_by_files=False, fetch_step=0.01, file_fraction=1, remake_weights=False, up_sample=True, weight_scale=1, max_resample=10, async_load=True, infinity_mode=False, in_memory=False, name=''):
        self._iters = {} if infinity_mode or in_memory else None
        _init_args = set(self.__dict__.keys())
        self._init_file_dict = file_dict
        self._init_load_range_and_fraction = load_range_and_fraction
        self._fetch_by_files = fetch_by_files
        self._fetch_step = fetch_step
        self._file_fraction = file_fraction
        self._async_load = async_load
        self._infinity_mode = infinity_mode
        self._in_memory = in_memory
        self._name = name
        self._sampler_options = {
            'up_sample': up_sample,
            'weight_scale': weight_scale,
            'max_resample': max_resample,
        }
        if for_training:
            self._sampler_options.update(training=True, shuffle=True, reweight=True)
        else:
            self._sampler_options.update(training=False, shuffle=False, reweight=False)
        if '.auto.yaml' in data_config_file:
            data_config_autogen_file = data_config_file
        else:
            data_config_md5 = _md5(data_config_file)
            data_config_autogen_file = data_config_file.replace('.yaml', '.%s.auto.yaml' % data_config_md5)
            import os
            if os.path.exists(data_config_autogen_file):
                data_config_file = data_config_autogen_file
        self._data_config = DataConfig.load(data_config_file)
        if for_training:
            if self._data_config._missing_standardization_info:
                from .preprocess import AutoStandardizer
                s = AutoStandardizer(file_dict, self._data_config)
                self._data_config = s.produce(data_config_autogen_file)
            if self._sampler_options['reweight'] and self._data_config.weight_name and not getattr(self._data_config, 'use_precomputed_weights', False):
                if remake_weights or self._data_config.reweight_hists is None:
                    from .preprocess import WeightMaker
                    w = WeightMaker(file_dict, self._data_config)
                    self._data_config = w.produce(data_config_autogen_file)
            if os.path.exists(data_config_autogen_file) and data_config_file != data_config_autogen_file:
                data_config_file = data_config_autogen_file
            self._data_config = DataConfig.load(data_config_file, load_observers=False, extra_selection=extra_selection)
        else:
            self._data_config = DataConfig.load(data_config_file, load_reweight_info=False, extra_test_selection=extra_selection)
        self._init_args = set(self.__dict__.keys()) - _init_args

    @property
    def config(self):
        return self._data_config

    def __iter__(self):
        if self._iters is None:
            kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
            return _SimpleIter(**kwargs)
        else:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            try:
                return self._iters[worker_id]
            except KeyError:
                kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
                self._iters[worker_id] = _SimpleIter(**kwargs)
                return self._iters[worker_id]
