import numpy as np
import math

def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    return np.concatenate(arrays, axis=axis)

def _stack(arrays, axis=1):
    if len(arrays) == 0:
        return np.array([])
    return np.stack(arrays, axis=axis)

def _pad(a, maxlen, value=0, dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = np.asarray(s)[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x

def _repeat_pad(a, maxlen, shuffle=False, dtype='float32'):
    x = np.concatenate([np.asarray(s) for s in a])
    x = np.tile(x, int(np.ceil(len(a) * maxlen / max(1, len(x)))))
    if shuffle:
        np.random.shuffle(x)
    x = x[:len(a) * maxlen].reshape((len(a), maxlen))
    mask = _pad([[] for _ in a], maxlen, value=1)
    x = _pad(a, maxlen) + mask * x
    return x.astype(dtype)

def _clip(a, a_min, a_max):
    return np.clip(a, a_min, a_max)

def _get_variable_names(expr, exclude=None):
    if exclude is None:
        exclude = ['awkward', 'ak', 'np', 'numpy', 'math', 'len']
    import ast
    root = ast.parse(expr)
    return sorted({node.id for node in ast.walk(root) if isinstance(node, ast.Name) and not node.id.startswith('_')} - set(exclude))

def _eval_expr(expr, table):

    '''
    _eval_expr 是一个函数，用于计算表达式的值。
    eg: a = _eval_expr('a + b', {'a': 1, 'b': 2}) -> 3
    '''
    tmp = {k: table[k] for k in _get_variable_names(expr)}
    tmp.update({'math': math, 'np': np, 'numpy': np, 'len': len})
    return eval(expr, tmp)
