from itertools import chain
import abc
from abc import abstractmethod
import numpy as np

#@title from fastNLP.core.samplers import *
class Sampler(object):
    r"""
    `Sampler` 类的基类. 规定以何种顺序取出data中的元素

    子类必须实现 ``__call__`` 方法. 输入 `DataSet` 对象, 返回其中元素的下标序列
    """
    
    def __call__(self, data_set):
        r"""
        :param DataSet data_set: `DataSet` 对象, 需要Sample的数据
        :return result: list(int) 其中元素的下标序列, ``data_set`` 中元素会按 ``result`` 中顺序取出
        """
        raise NotImplementedError

class SequentialSampler(Sampler):
    r"""
    顺序取出元素的 `Sampler`

    """
    
    def __call__(self, data_set):
        return list(range(len(data_set)))

class RandomSampler(Sampler):
    r"""
    随机化取元素的 `Sampler`

    """
    
    def __call__(self, data_set):
        return list(np.random.permutation(len(data_set)))


class SortedSampler(Sampler):
    r"""
    按照sample的长度进行排序，主要在测试的时候使用，可以加速测试（因为减少了padding）
    """
    def __init__(self, seq_len_field_name='seq_len', descending=True):
        """

        :param str seq_len_field_name: 按哪个field进行排序。如果传入的field是数字，则直接按照该数字大小排序；如果传入的field不是
            数字，则使用该field的长度进行排序
        :param bool descending: 是否降序排列
        """
        self.seq_len_field_name = seq_len_field_name
        self.descending = descending


    def __call__(self, data_set):
        seq_lens = data_set.get_field(self.seq_len_field_name).content
        try:
            seq_lens = list(map(len, seq_lens))
        except:
            pass

        orders = np.argsort(seq_lens).tolist()  # 从小到大的顺序
        if self.descending:
            orders = orders[::-1]
        return orders

class BucketSampler(Sampler):
    r"""
    带Bucket的 `Random Sampler`. 可以随机地取出长度相似的元素
    """
    
    def __init__(self, num_buckets=10, batch_size=None, seq_len_field_name='seq_len'):
        r"""
        
        :param int num_buckets: bucket的数量
        :param int batch_size: batch的大小. 默认为None，Trainer/Tester在调用BucketSampler时，会将该值正确设置，如果是非
            Trainer/Tester场景使用，需要显示传递该值
        :param str seq_len_field_name: 对应序列长度的 `field` 的名字
        """
        self.num_buckets = num_buckets
        self.batch_size = batch_size
        self.seq_len_field_name = seq_len_field_name


    def set_batch_size(self, batch_size):
        r"""

        :param int batch_size: 每个batch的大小
        :return:
        """
        self.batch_size = batch_size


    def __call__(self, data_set):
        if self.batch_size is None:
            raise RuntimeError("batch_size is None.")
        seq_lens = data_set.get_all_fields()[self.seq_len_field_name].content
        total_sample_num = len(seq_lens)
        
        bucket_indexes = []
        assert total_sample_num >= self.num_buckets, "The number of samples is smaller than the number of buckets."
        num_sample_per_bucket = total_sample_num // self.num_buckets
        for i in range(self.num_buckets):
            bucket_indexes.append([num_sample_per_bucket * i, num_sample_per_bucket * (i + 1)])
        bucket_indexes[-1][1] = total_sample_num
        
        sorted_seq_lens = list(sorted([(idx, seq_len) for
                                       idx, seq_len in zip(range(total_sample_num), seq_lens)],
                                      key=lambda x: x[1]))
        
        batchs = []
        
        left_init_indexes = []
        for b_idx in range(self.num_buckets):
            start_idx = bucket_indexes[b_idx][0]
            end_idx = bucket_indexes[b_idx][1]
            sorted_bucket_seq_lens = sorted_seq_lens[start_idx:end_idx]
            left_init_indexes.extend([tup[0] for tup in sorted_bucket_seq_lens])
            num_batch_per_bucket = len(left_init_indexes) // self.batch_size
            np.random.shuffle(left_init_indexes)
            for i in range(num_batch_per_bucket):
                batchs.append(left_init_indexes[i * self.batch_size:(i + 1) * self.batch_size])
            left_init_indexes = left_init_indexes[num_batch_per_bucket * self.batch_size:]
        if (left_init_indexes) != 0:
            batchs.append(left_init_indexes)
        np.random.shuffle(batchs)
        
        return list(chain(*batchs))


class ConstTokenNumSampler(Sampler):
    """
    尽量保证每个batch的输入token数量是接近的。

    使用示例
    >>> # 假设已经有了tr_data并有一个field叫做seq_len保存了每个instance的token数量
    >>> from fastNLP import DataSetIter, Trainer
    >>> sampler = ConstTokenNumSampler('src_seq_len', max_token=4096)
    >>>
    >>> # 直接将sampler传入Trainer中，此时batch_size参数的值会被忽略
    >>> trainer = Trainer(tr_data, model, optimizer=optimizer, loss=TranslationLoss(),
    >>>             batch_size=1, sampler=sampler, drop_last=False, update_every=1)
    """
    def __init__(self, seq_len_field_name, max_token=4096, max_sentence=-1, need_be_multiple_of=1, num_bucket=-1):
        """

        :param List[int] seq_len_field_name: 哪个field指示的sample的长度
        :param int max_token: 每个batch的最大的token数量
        :param int max_sentence: 每个batch最多多少个instance, -1表示根据max_token决定
        :param int need_be_multiple_of: 生成的batch的instance的数量需要是几的倍数，在DataParallel场景下会用到
        :param int num_bucket: 将数据按长度拆分为num_bucket个bucket，batch中的sample尽量在bucket之中进行组合，这样可以减少padding。
        """
        assert (max_sentence!=-1 and max_sentence>=need_be_multiple_of) or max_sentence<1
        self.seq_len_field_name = seq_len_field_name
        self.num_bucket = num_bucket
        self.max_token = max_token
        self._max_sentence = max_sentence
        self.need_be_multiple_of = need_be_multiple_of

    def __call__(self, data_set):
        assert len(data_set)>self.num_bucket, "The number of samples should be larger than buckets."
        seq_len = data_set.get_field(self.seq_len_field_name)
        self.seq_len = seq_len
        seq_len_indice = [(length, i) for i, length in enumerate(seq_len)]
        seq_len_indice.sort(key=lambda x: x[0])
        indice_in_buckets = []
        if self.num_bucket>0:
            sample_per_bucket = len(seq_len_indice)//self.num_bucket
            i = 0
            while len(indice_in_buckets)<len(seq_len_indice):
                indice_in_buckets.append(seq_len_indice[i*sample_per_bucket:(i+1)*sample_per_bucket])
                i += 1
        else:
            indice_in_buckets = [seq_len_indice]
        self.indice_in_buckets = indice_in_buckets
        self.get_new_order()

    @property
    def max_sentence(self):
        if self._max_sentence<1:
            return 100000000
        return self._max_sentence

    @max_sentence.setter
    def max_sentence(self, max_sentence):
        self._max_sentence = max_sentence

    def get_new_order(self):
        np.random.shuffle(self.indice_in_buckets)
        for bucket in self.indice_in_buckets:
            np.random.shuffle(bucket)
        indices = list(chain(*self.indice_in_buckets))
        batches = []
        cur_max_len = 0
        batch = []
        for length, i in indices:
            max_len = max(length, cur_max_len)
            if max_len*(len(batch)+1)>self.max_token or len(batch)>=self.max_sentence:
                left_sample = len(batch) % self.need_be_multiple_of
                add_samples = batch.copy()
                cur_max_len =length
                if left_sample!=0:
                    add_samples = add_samples[:-left_sample]
                    batch = batch[-left_sample:]
                    cur_max_len = max(cur_max_len, max(batch))
                else:
                    batch = []
                if len(add_samples)==0:
                    raise RuntimeError(f"The sample `{i}` is too long to make a batch with {self.need_be_multiple_of} samples.")
                batches.append(add_samples)
            else:
                cur_max_len = max_len
            batch.append(i)
        if batch:
            left_sample = len(batch) % self.need_be_multiple_of
            add_samples = batch.copy()
            if left_sample != 0:
                add_samples = add_samples[:-left_sample].copy()
            if add_samples:
                batches.append(add_samples)
        np.random.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch
        self.get_new_order()

    def __len__(self):
        return len(self.batches)

#@title fastNLP.core.batch

class SamplerAdapter(torch.utils.data.Sampler):
    r"""
    用于传入torch.utils.data.DataLoader中，DataLoader会调用__iter__()方法获取index(一次只取一个int)

    """
    def __init__(self, sampler, dataset):
        super().__init__(dataset)
        self.sampler = sampler
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.sampler(self.dataset))

class BatchIter:
    r"""
    Trainer用于迭代数据的类。继承该类，并实现get_num_batches(), get_batch_indices(), num_batches(), __iter__()方法以及dataset属性。

    """
    def __init__(self, dataset, batch_size=1, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None,
                 batch_sampler=None):
        if isinstance(sampler, Sampler):  # 如果时fastNLP的sampler需要adapt一下
            sampler = SamplerAdapter(sampler=sampler or SequentialSampler(), dataset=dataset)
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        # DataLoader的collate_fn输入是List[]，里面的元素是dataset[index]返回的结果
        if collate_fn is None:
            # pytoch <= 1.1 中不能设置collate_fn=None
            self.dataiter = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size, sampler=self.sampler,
                num_workers=num_workers,
                pin_memory=pin_memory, drop_last=drop_last,
                timeout=timeout, worker_init_fn=worker_init_fn,
                batch_sampler=batch_sampler)
        else:
            self.dataiter = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size, sampler=self.sampler,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory, drop_last=drop_last,
                timeout=timeout, worker_init_fn=worker_init_fn,
                batch_sampler=batch_sampler)

        # 以sampler的数量为准，因为DistributedSampler的时候每个进程上并不是所有的数据都用上了
        if self.batch_sampler is None:
            self._num_batches = self.get_num_batches(len(self.dataiter.sampler), batch_size, drop_last)
        else:
            self._num_batches = len(self.batch_sampler)
        self.batch_size = batch_size
        self.cur_batch_indices = None

    @property
    def num_batches(self):
        return self._num_batches

    @num_batches.setter
    def num_batches(self, value):
        self._num_batches = value

    def init_iter(self):
        pass

    @staticmethod
    def get_num_batches(num_samples, batch_size, drop_last):
        r"""
        计算batch的数量。用于前端显示进度

        :param int num_samples:
        :param int batch_size:
        :param bool drop_last: 如果最后一个batch没有batch_size这么多，是否就丢掉。
        :return:
        """
        num_batches = num_samples // batch_size
        if not drop_last and (num_samples % batch_size > 0):
            num_batches += 1
        return num_batches


    def get_batch_indices(self):
        r"""
        获取最近输出的batch的index。用于溯源当前batch的数据

        :return:
        """
        return self.cur_batch_indices


    def __len__(self):
        return self.num_batches

    @property
    def dataset(self):
        r"""
        获取正在参与iterate的dataset

        :return:
        """
        return self.dataiter.dataset

    @abstractmethod
    def __iter__(self):
        r"""
        用于实际数据循环的类，返回值需要为两个dict, 第一个dict中的内容会认为是input, 第二个dict中的内容会认为是target

        :return:
        """
        raise NotImplemented

def _to_tensor(batch, field_dtype):
    r"""

    :param batch: np.array()
    :param field_dtype: 数据类型
    :return: batch, flag. 如果传入的数据支持转为tensor，返回的batch就是tensor，且flag为True；如果传入的数据不支持转为tensor，
        返回的batch就是原来的数据，且flag为False
    """
    try:
        if field_dtype is not None and isinstance(field_dtype, type)\
                and issubclass(field_dtype, Number) \
                and not isinstance(batch, torch.Tensor):
            new_batch = torch.as_tensor(batch)
            flag = True
        else:
            new_batch = batch
            flag = False
        if torch.is_tensor(new_batch):
            if 'float' in new_batch.dtype.__repr__():
                new_batch = new_batch.float()
            elif 'int' in new_batch.dtype.__repr__():
                new_batch = new_batch.long()
        return new_batch, flag
    except Exception as e:
        raise e

def _pad(batch_dict, dataset, as_numpy):
    result = {}
    for n, vlist in batch_dict.items():
        f = dataset.field_arrays[n]
        if f.padder is None:
            result[n] = np.array(vlist)
        else:
            res = f.pad(vlist)
            if not as_numpy:
                res, _ = _to_tensor(res, field_dtype=f.dtype)
            result[n] = res

    return result

class DataSetGetter:
    r"""
    传递给torch.utils.data.DataLoader获取数据，DataLoder会传入int的idx获取数据(调用这里的__getitem__()函数)。
    """
    def __init__(self, dataset: DataSet, as_numpy=False):
        self.dataset = dataset
        self.as_numpy = as_numpy
        self.idx_list = list(range(len(dataset)))

        self.x_names = {n for n, f in dataset.get_all_fields().items() if f.is_input}
        self.y_names = {n for n, f in dataset.get_all_fields().items() if f.is_target}

    def __getitem__(self, idx: int):
        # mapping idx to sampled idx
        idx = self.idx_list[idx]
        ins = self.dataset[idx]
        return idx, ins

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, ins_list: list):
        r"""

        :param batch: [[idx1, x_dict1, y_dict1], [idx2, x_dict2, y_dict2], [xx, xx, xx]]
        :return:
        """
        indices = []
        sin_x, sin_y = defaultdict(list), defaultdict(list)
        # 收集需要关注的field的数据
        for idx, ins in ins_list:
            indices.append(idx)
            for n, v in ins.items():
                if n in self.x_names:
                    sin_x[n].append(v)
                if n in self.y_names:
                    sin_y[n].append(v)
        # 根据情况，进行pad
        sin_x = _pad(sin_x, dataset=self.dataset, as_numpy=self.as_numpy)
        sin_y = _pad(sin_y, dataset=self.dataset, as_numpy=self.as_numpy)

        if not self.dataset.collater.is_empty():
            bx, by = self.dataset._collate_batch(ins_list)
            sin_x.update(bx)
            sin_y.update(by)

        return indices, sin_x, sin_y

    def __getattr__(self, item):
        if hasattr(self.dataset, item):
            return getattr(self.dataset, item)
        else:
            raise AttributeError("'DataSetGetter' object has no attribute '{}'".format(item))

class DataSetIter(BatchIter):
    r"""
    DataSetIter 用于从 `DataSet` 中按一定的顺序, 依次按 ``batch_size`` 的大小将数据取出，通过使用DataSetIter，可以不需要考虑
        输入的padding(由DataSet中每列的Padder决定了)以及不需要考虑将数据转为tensor。
    组成 `x` 和 `y`::

        batch = DataSetIter(data_set, batch_size=16, sampler=SequentialSampler())
        num_batch = len(batch)
        for batch_x, batch_y in batch:
            # do stuff ...

    """
    def __init__(self, dataset, batch_size=1, sampler=None, as_numpy=False, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=None):
        r"""
        
        :param dataset: :class:`~fastNLP.DataSet` 对象, 数据集
        :param int batch_size: 取出的batch大小
        :param sampler: 规定使用的 :class:`~fastNLP.Sampler` 方式. 若为 ``None`` , 使用 :class:`~fastNLP.SequentialSampler`.
    
            Default: ``None``
        :param bool as_numpy: 若为 ``True`` , 输出batch为 numpy.array. 否则为 :class:`torch.Tensor`.

            Default: ``False``
        :param int num_workers: 使用多少个进程来预处理数据
        :param bool pin_memory: 是否将产生的tensor使用pin memory, 可能会加快速度。
        :param bool drop_last: 如果最后一个batch没有batch_size这么多sample，就扔掉最后一个
        :param timeout: 生成一个batch的timeout值
        :param worker_init_fn: 在每个worker启动时调用该函数，会传入一个值，该值是worker的index。
        :param batch_sampler: 当每次batch取出的数据数量不一致时，可以使用该sampler。batch_sampler每次iter应该输出一个list的index。
            当batch_sampler不为None时，参数batch_size, sampler, drop_last会被忽略。
        """
        assert isinstance(dataset, DataSet)
        dataset = DataSetGetter(dataset, as_numpy)
        collate_fn = dataset.collate_fn
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
            drop_last = False
        super().__init__(
            dataset=dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            collate_fn=collate_fn, batch_sampler=batch_sampler
        )


    def __iter__(self):
        self.init_iter()
        for indices, batch_x, batch_y in self.dataiter:
            self.cur_batch_indices = indices
            yield batch_x, batch_y