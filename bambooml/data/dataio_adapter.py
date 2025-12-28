import numpy as np

class stack_groups:
    '''
    stack_groups 是一个类，用于将多个数组堆叠在一起。
    stack_groups 类包含两个静态方法：pad_array 和 repeat_pad。
    pad_array 方法用于将一个数组填充到指定长度。
    repeat_pad 方法生成一个形状为 (batch_size, maxlen) 的循环序列矩阵，
覆盖原始数据，用于模拟重复或循环序列填充。

    静态方法：不需要实例化类，可以直接使用类名调用。和类方法的区别是：静态方法不需要 self 参数，而类方法需要 self 参数。且静态方法不能访问类的属性。
    '''
    @staticmethod
    def pad_array(a, maxlen, value=0):
        x = (np.ones((len(a), maxlen)) * value).astype('float32')
        for idx, s in enumerate(a): 
            s = np.asarray(s)
            if s.size == 0:
                continue
            trunc = s[:maxlen].astype('float32')  # 如果s的长度小于maxlen，s[:maxlen]会返回s的所有元素，否则返回s的前maxlen个元素。
            x[idx, :len(trunc)] = trunc
        return x

    @staticmethod
    def repeat_pad(a, maxlen):
        x = np.concatenate([np.asarray(s) for s in a])  
        x = np.tile(x, int(np.ceil(len(a) * maxlen / max(1, len(x)))))
        x = x[:len(a) * maxlen].reshape((len(a), maxlen))
        mask = stack_groups.pad_array([[] for _ in a], maxlen, value=1)
        return stack_groups.pad_array(a, maxlen) + mask * x
