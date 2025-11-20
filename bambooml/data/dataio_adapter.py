import numpy as np

class stack_groups:
    @staticmethod
    def pad_array(a, maxlen, value=0):
        x = (np.ones((len(a), maxlen)) * value).astype('float32')
        for idx, s in enumerate(a):
            s = np.asarray(s)
            if s.size == 0:
                continue
            trunc = s[:maxlen].astype('float32')
            x[idx, :len(trunc)] = trunc
        return x

    @staticmethod
    def repeat_pad(a, maxlen):
        x = np.concatenate([np.asarray(s) for s in a])
        x = np.tile(x, int(np.ceil(len(a) * maxlen / max(1, len(x)))))
        x = x[:len(a) * maxlen].reshape((len(a), maxlen))
        mask = stack_groups.pad_array([[] for _ in a], maxlen, value=1)
        return stack_groups.pad_array(a, maxlen) + mask * x
