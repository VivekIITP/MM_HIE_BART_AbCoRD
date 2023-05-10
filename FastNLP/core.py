import _pickle
from copy import deepcopy
from numbers import Number
import inspect
import numpy as np
import abc
from abc import abstractmethod
from prettytable import PrettyTable
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from tqdm.auto import tqdm
from fastNLP import logger

#@title fastNLP.core.instance
class Instance(object):
    r"""
    Instance是fastNLP中对应一个sample的类。每个sample在fastNLP中是一个Instance对象。
    Instance一般与 :class:`~fastNLP.DataSet` 一起使用, Instance的初始化如下面的Example所示::
    
        >>>from fastNLP import Instance
        >>>ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2])
        >>>ins["field_1"]
        [1, 1, 1]
        >>>ins.add_field("field_3", [3, 3, 3])
        >>>ins = Instance(**{'x1': 1, 'x2':np.zeros((3, 4))})
    """

    def __init__(self, **fields):

        self.fields = fields

    def add_field(self, field_name, field):
        r"""
        向Instance中增加一个field

        :param str field_name: 新增field的名称
        :param Any field: 新增field的内容
        """
        self.fields[field_name] = field


    def items(self):
        r"""
        返回一个迭代器，迭代器返回两个内容，第一个内容是field_name, 第二个内容是field_value
        
        :return: 一个迭代器
        """
        return self.fields.items()


    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, name):
        if name in self.fields:
            return self.fields[name]
        else:
            raise KeyError("{} not found".format(name))

    def __setitem__(self, name, field):
        return self.add_field(name, field)

    def __repr__(self):
        return str(pretty_table_printer(self))

#@title fastNLP.core.fields

class SetInputOrTargetException(Exception):
    def __init__(self, msg, index=None, field_name=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了
        self.field_name = field_name  # 标示当前field的名称


class AppendToTargetOrInputException(Exception):
    def __init__(self, msg, index=None, field_name=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了
        self.field_name = field_name  # 标示当前field的名称


def _is_iterable(value):
    # 检查是否是iterable的, duck typing
    try:
        iter(value)
        return True
    except BaseException as e:
        return False


def _get_ele_type_and_dim(cell: Any, dim=0):
    r"""
    识别cell的类别与dimension的数量

    numpy scalar type:https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
    :param cell:
    :param dim:
    :return:
    """
    if isinstance(cell, (str, Number, np.bool_)):
        if hasattr(cell, 'dtype'):
            return cell.dtype.type, dim
        return type(cell), dim
    elif isinstance(cell, list):
        dim += 1
        res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    elif isinstance(cell, torch.Tensor):
        return cell.dtype, cell.dim() + dim  # 如果是torch.mean的结果是0
    elif isinstance(cell, np.ndarray):
        if cell.dtype != np.dtype('O'):  # 如果不是object的话说明是well-formatted的了
            return cell.dtype.type, cell.ndim + dim  # dtype.type返回的会是np.int32, np.float等
        # 否则需要继续往下iterate
        dim += 1
        res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    else:  # 包含tuple, set, dict以及其它的类型
        raise SetInputOrTargetException(f"Cannot process type:{type(cell)}.")


class Padder:
    r"""
    所有padder都需要继承这个类，并覆盖__call__方法。
    用于对batch进行padding操作。传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前deepcopy一份。

    .. py:function:: __call__(self, contents, field_name, field_ele_dtype):
    
    """
    
    def __init__(self, pad_val=0, **kwargs):
        r"""
        
        :param List[Any] contents: 传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前
            deepcopy一份。
        :param str, field_name: field的名称。
        :param np.int64,np.float64,np.str,None, field_ele_dtype: 该field的内层元素的类型。如果该field的ignore_type为True，该这个值为None。
        :return: np.array([padded_element])
        """
        self.pad_val = pad_val

    
    def set_pad_val(self, pad_val):
        self.pad_val = pad_val

    def get_pad_val(self):
        return self.pad_val

    @abstractmethod
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        r"""
        传入的是List内容。假设有以下的DataSet。

        :param List[Any] contents: 传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前
            deepcopy一份。
        :param str, field_name: field的名称。
        :param np.int64,np.float64,np.str,None, field_ele_dtype: 该field的内层元素的类型。如果该field的ignore_type为True，
            该这个值为None。
        :param dim: 这个field的维度。当ignore_type为True时，该值为None
        :return: np.array([padded_element])

        Example::

            from fastNLP import DataSet
            from fastNLP import Instance
            dataset = DataSet()
            dataset.append(Instance(sent='this is a demo', length=4,
                                    chars=[['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']]))
            dataset.append(Instance(sent='another one', length=2,
                                    chars=[['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]))
            如果调用
            batch = dataset.get([0,1], pad=True)
            sent这个field的padder的__call__会接收到的内容会是
                [
                    'this is a demo',
                    'another one'
                ]

            length这个field的padder的__call__会接收到的内容会是
                [4, 2]

            chars这个field的padder的__call__会接收到的内容会是
                [
                    [['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']],
                    [['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]
                ]

        即把每个instance中某个field的内容合成一个List传入

        """
        raise NotImplementedError

class AutoPadder(Padder):
    r"""
    根据contents的数据自动判定是否需要做padding。

    1 如果元素类型(元素类型是指field中最里层元素的数据类型, 可以通过FieldArray.dtype查看，比如['This', 'is', ...]的元素类
    型为str, [[1,2], ...]的元素类型为int)的数据不为数值类型则不会进行pad

    2 如果元素类型为数值类型,比如np.int64, np.float64, int, float, torch.int64等

        2.1 如果该field的内容为数值类型(包括int, float等)，比如为seq_len, 则不进行padding

        2.2 如果该field的内容等价于一维list, 那么会将Batch中的List pad为一样长。

        2.3 如果该field的内容等价于二维list，那么会按照英语character padding的方式进行padding。如果是character padding建议使用
            :class: fastNLP.EngChar2DPadder.

        2.4 如果该field的内容等价于三维list，则如果每个instance在每个维度上相等，会组成一个batch的tensor返回，这种情况应该是为图片
            的情况。

    3 其它情况不进行处理，返回一个np.array类型。
    """
    
    def __init__(self, pad_val=0):
        super().__init__(pad_val=pad_val)
    
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        if field_ele_dtype:
            if dim > 3:
                return np.array(contents)
            if isinstance(field_ele_dtype, type) and \
                    (issubclass(field_ele_dtype, np.number) or issubclass(field_ele_dtype, Number)):
                if dim == 0:
                    array = np.array(contents, dtype=field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        array[i, :len(content_i)] = content_i
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    array = np.full((len(contents), max_len, max_word_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            array[i, j, :len(content_ii)] = content_ii
                else:
                    shape = np.shape(contents)
                    if len(shape) == 4:  # 说明各dimension是相同的大小
                        array = np.array(contents, dtype=field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return array
            elif str(field_ele_dtype).startswith('torch'):
                if dim == 0:
                    tensor = torch.tensor(contents).to(field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    tensor = torch.full((len(contents), max_len), fill_value=self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        tensor[i, :len(content_i)] = content_i.clone().detach()
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    tensor = torch.full((len(contents), max_len, max_word_len), fill_value=self.pad_val,
                                        dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
                else:
                    shapes = set([np.shape(content_i) for content_i in contents])
                    if len(shapes) > 1:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                    shape = shapes.pop()
                    if len(shape) == 3:
                        tensor = torch.full([len(contents)] + list(shape), fill_value=self.pad_val,
                                            dtype=field_ele_dtype)
                        for i, content_i in enumerate(contents):
                            tensor[i] = content_i.clone().detach().to(field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return tensor
            else:
                return np.array(contents)  # 不进行任何操作
        else:
            return np.array(contents)


class FieldArray:
    def __init__(self, name, content, is_target=False, is_input=False, padder=AutoPadder(), ignore_type=False,
                 use_1st_ins_infer_dim_type=True):
        if len(content) == 0:
            raise RuntimeError("Empty fieldarray is not allowed.")
        _content = content
        try:
            _content = list(_content)
        except BaseException as e:
            logger.error(f"Cannot convert content(of type:{type(content)}) into list.")
            raise e
        self.name = name
        self.content = _content
        self._ignore_type = ignore_type
        #  根据input的情况设置input，target等
        self._cell_ndim = None  # 多少维度， 如果value是1, dim为0; 如果value是[1, 2], dim=2
        self.dtype = None  # 最内层的element都是什么类型的
        self._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
        self._is_input = False
        self._is_target = False

        if is_input:
            self.is_input = is_input
        if is_target:
            self.is_target = is_target

        self.set_padder(padder)

    @property
    def ignore_type(self):
        return self._ignore_type

    @ignore_type.setter
    def ignore_type(self, value):
        if value:
            self._cell_ndim = None
            self.dtype = None
        self._ignore_type = value

    @property
    def is_input(self):
        return self._is_input

    @is_input.setter
    def is_input(self, value):
        r"""
            当 field_array.is_input = True / False 时被调用
        """
        # 如果(value为True)且(_is_input和_is_target都是False)且(ignore_type为False)
        if value is True and \
                self._is_target is False and \
                self._ignore_type is False:
            self._check_dtype_and_ndim(only_check_1st_ins_dim_type=self._use_1st_ins_infer_dim_type)
        if value is False and self._is_target is False:
            self.dtype = None
            self._cell_ndim = None
        self._is_input = value

    @property
    def is_target(self):
        return self._is_target

    @is_target.setter
    def is_target(self, value):
        r"""
        当 field_array.is_target = True / False 时被调用
        """
        if value is True and \
                self._is_input is False and \
                self._ignore_type is False:
            self._check_dtype_and_ndim(only_check_1st_ins_dim_type=self._use_1st_ins_infer_dim_type)
        if value is False and self._is_input is False:
            self.dtype = None
            self._cell_ndim = None
        self._is_target = value

    def _check_dtype_and_ndim(self, only_check_1st_ins_dim_type=True):
        r"""
        检查当前content所有的element是否是同一个类型，且是否每个元素具有相同的维度。通过的话，设置_cell_ndim与_ele_type属性；没有
            通过将直接报错.

        :param bool only_check_1st_ins_dim_type: 是否只检查第一个元素的type和dim
        :return:
        """
        cell_0 = self.content[0]
        index = 0
        try:
            type_0, dim_0 = _get_ele_type_and_dim(cell_0)
            if not only_check_1st_ins_dim_type:
                for cell in self.content[1:]:
                    index += 1
                    type_i, dim_i = _get_ele_type_and_dim(cell)
                    if type_i != type_0:
                        raise SetInputOrTargetException(
                            "Type:{} in index {} is different from the first element with type:{}."
                            ".".format(type_i, index, type_0))
                    if dim_0 != dim_i:
                        raise SetInputOrTargetException(
                            "Dimension:{} in index {} is different from the first element with "
                            "dimension:{}.".format(dim_i, index, dim_0))
            self._cell_ndim = dim_0
            self.dtype = type_0
        except SetInputOrTargetException as e:
            e.index = index
            raise e

    def append(self, val: Any):
        r"""
        :param val: 把该val append到fieldarray。
        :return:
        """
        if (self._is_target or self._is_input) and self._ignore_type is False and not self._use_1st_ins_infer_dim_type:
            type_, dim_ = _get_ele_type_and_dim(val)
            if self.dtype != type_:
                raise AppendToTargetOrInputException(f"Value(type:{type_}) are of different types with "
                                                     f"previous values(type:{self.dtype}).")
            if self._cell_ndim != dim_:
                raise AppendToTargetOrInputException(f"Value(dim:{dim_}) are of different dimensions with "
                                                     f"previous values(dim:{self._cell_ndim}).")
            self.content.append(val)
        else:
            self.content.append(val)

    def pop(self, index):
        r"""
        删除该field中index处的元素
        :param int index: 从0开始的数据下标。
        :return:
        """
        self.content.pop(index)

    def __getitem__(self, indices):
        return self.get(indices, pad=False)

    def __setitem__(self, idx, val):
        assert isinstance(idx, int)
        if (self._is_target or self._is_input) and self.ignore_type is False:  # 需要检测类型
            type_, dim_ = _get_ele_type_and_dim(val)
            if self.dtype != type_:
                raise RuntimeError(f"Value(type:{type_}) are of different types with "
                                   f"other values(type:{self.dtype}).")
            if self._cell_ndim != dim_:
                raise RuntimeError(f"Value(dim:{dim_}) are of different dimensions with "
                                   f"previous values(dim:{self._cell_ndim}).")
        self.content[idx] = val

    def get(self, indices, pad=True):
        r"""
        根据给定的indices返回内容。

        :param int,List[int] indices: 获取indices对应的内容。
        :param bool pad: 是否对返回的结果进行padding。仅对: (1) indices为List[int]; (2)padder不为None; (3)field设置了input
            或target，有效
        :return: 根据给定的indices返回的内容，可能是单个值或ndarray
        """
        if isinstance(indices, int):
            return self.content[indices]

        contents = [self.content[i] for i in indices]
        if self.padder is None or pad is False:
            return np.array(contents)
        elif self.is_input or self.is_target:
            return self.pad(contents)
        else:
            return np.array(contents)

    def pad(self, contents):
        r"""
        传入list的contents，将contents使用padder进行padding，contents必须为从本FieldArray中取出的。

        :param list contents:
        :return:
        """
        return self.padder(contents, field_name=self.name, field_ele_dtype=self.dtype, dim=self._cell_ndim)

    def set_padder(self, padder):
        r"""
        设置padder，在这个field进行pad的时候用这个padder进行pad，如果为None则不进行pad。

        :param padder: :class:`~fastNLP.Padder` 类型，设置为None即删除padder。
        """
        if padder is not None:
            assert isinstance(padder, Padder), "padder must be of type `fastNLP.core.Padder`."
            self.padder = deepcopy(padder)
        else:
            self.padder = None

    def set_pad_val(self, pad_val):
        r"""
        修改padder的pad_val.

        :param int pad_val: 该field的pad值设置为该值。
        """
        if self.padder is not None:
            self.padder.set_pad_val(pad_val)
        return self

    def __len__(self):
        r"""
        Returns the size of FieldArray.

        :return int length:
        """
        return len(self.content)

    def to(self, other):
        r"""
        将other的属性复制给本FieldArray(other必须为FieldArray类型).
        属性包括 is_input, is_target, padder, ignore_type

        :param  other: :class:`~fastNLP.FieldArray` 从哪个field拷贝属性
        :return: :class:`~fastNLP.FieldArray`
        """
        assert isinstance(other, FieldArray), "Only supports fastNLP.FieldArray type, not {}.".format(type(other))

        self.ignore_type = other.ignore_type
        self.is_input = other.is_input
        self.is_target = other.is_target
        self.padder = other.padder

        return self

    def split(self, sep: str = None, inplace: bool = True):
        r"""
        依次对自身的元素使用.split()方法，应该只有当本field的元素为str时，该方法才有用。将返回值

        :param sep: 分割符，如果为None则直接调用str.split()。
        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
        :return: List[List[str]] or self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                new_contents.append(cell.split(sep))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def int(self, inplace: bool = True):
        r"""
        将本field中的值调用int(cell). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([int(value) for value in cell])
                else:
                    new_contents.append(int(cell))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def float(self, inplace=True):
        r"""
        将本field中的值调用float(cell). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
        :return:
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([float(value) for value in cell])
                else:
                    new_contents.append(float(cell))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def bool(self, inplace=True):
        r"""
        将本field中的值调用bool(cell). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
        :return:
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([bool(value) for value in cell])
                else:
                    new_contents.append(bool(cell))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e

        return self._after_process(new_contents, inplace=inplace)

    def lower(self, inplace=True):
        r"""
        将本field中的值调用cell.lower(). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([value.lower() for value in cell])
                else:
                    new_contents.append(cell.lower())
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def upper(self, inplace=True):
        r"""
        将本field中的值调用cell.lower(). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([value.upper() for value in cell])
                else:
                    new_contents.append(cell.upper())
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def value_count(self):
        r"""
        返回该field下不同value的数量。多用于统计label数量

        :return: Counter, key是label，value是出现次数
        """
        count = Counter()

        def cum(cell):
            if _is_iterable(cell) and not isinstance(cell, str):
                for cell_ in cell:
                    cum(cell_)
            else:
                count[cell] += 1

        for cell in self.content:
            cum(cell)
        return count

    def _after_process(self, new_contents, inplace):
        r"""
        当调用处理函数之后，决定是否要替换field。

        :param new_contents:
        :param inplace:
        :return: self或者生成的content
        """
        if inplace:
            self.content = new_contents
            try:
                self.is_input = self.is_input
                self.is_target = self.is_input
            except SetInputOrTargetException as e:
                logger.error("The newly generated field cannot be set as input or target.")
                raise e
            return self
        else:
            return new_contents

#@title fastNLP.core.Collater

class Collater:
    r"""
    辅助DataSet管理collate_fn的类

    """
    def __init__(self):
        self.collate_fns = {}

    def add_fn(self, fn, name=None):
        r"""
        向collater新增一个collate_fn函数

        :param callable fn:
        :param str,int name:
        :return:
        """
        if name in self.collate_fns:
            logger.warn(f"collate_fn:{name} will be overwritten.")
        if name is None:
            name = len(self.collate_fns)
        self.collate_fns[name] = fn

    def is_empty(self):
        r"""
        返回是否包含collate_fn

        :return:
        """
        return len(self.collate_fns) == 0

    def delete_fn(self, name=None):
        r"""
        删除collate_fn

        :param str,int name: 如果为None就删除最近加入的collate_fn
        :return:
        """
        if not self.is_empty():
            if name in self.collate_fns:
                self.collate_fns.pop(name)
            elif name is None:
                last_key = list(self.collate_fns.keys())[0]
                self.collate_fns.pop(last_key)

    def collate_batch(self, ins_list):
        bx, by = {}, {}
        for name, fn in self.collate_fns.items():
            try:
                batch_x, batch_y = fn(ins_list)
            except BaseException as e:
                logger.error(f"Exception:`{e}` happens when call collate_fn:`{name}`.")
                raise e
            bx.update(batch_x)
            by.update(batch_y)
        return bx, by

    def copy_from(self, col):
        assert isinstance(col, Collater)
        new_col = Collater()
        new_col.collate_fns = deepcopy(col.collate_fns)
        return new_col
    r"""
    辅助DataSet管理collate_fn的类

    """
    def __init__(self):
        self.collate_fns = {}

    def add_fn(self, fn, name=None):
        r"""
        向collater新增一个collate_fn函数

        :param callable fn:
        :param str,int name:
        :return:
        """
        if name in self.collate_fns:
            logger.warn(f"collate_fn:{name} will be overwritten.")
        if name is None:
            name = len(self.collate_fns)
        self.collate_fns[name] = fn

    def is_empty(self):
        r"""
        返回是否包含collate_fn

        :return:
        """
        return len(self.collate_fns) == 0

    def delete_fn(self, name=None):
        r"""
        删除collate_fn

        :param str,int name: 如果为None就删除最近加入的collate_fn
        :return:
        """
        if not self.is_empty():
            if name in self.collate_fns:
                self.collate_fns.pop(name)
            elif name is None:
                last_key = list(self.collate_fns.keys())[0]
                self.collate_fns.pop(last_key)

    def collate_batch(self, ins_list):
        bx, by = {}, {}
        for name, fn in self.collate_fns.items():
            try:
                batch_x, batch_y = fn(ins_list)
            except BaseException as e:
                logger.error(f"Exception:`{e}` happens when call collate_fn:`{name}`.")
                raise e
            bx.update(batch_x)
            by.update(batch_y)
        return bx, by

    def copy_from(self, col):
        assert isinstance(col, Collater)
        new_col = Collater()
        new_col.collate_fns = deepcopy(col.collate_fns)
        return new_col

#@title fastNLP.core.Dataset

class ApplyResultException(Exception):
    def __init__(self, msg, index=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index

class DataSet(object):
    r"""
    fastNLP的数据容器，详细的使用方法见文档  :mod:`fastNLP.core.dataset`
    """

    def __init__(self, data=None):
        r"""
        
        :param data: 如果为dict类型，则每个key的value应该为等长的list; 如果为list，
            每个元素应该为具有相同field的 :class:`~fastNLP.Instance` 。
        """
        self.field_arrays = {}
        if data is not None:
            if isinstance(data, dict):
                length_set = set()
                for key, value in data.items():
                    length_set.add(len(value))
                assert len(length_set) == 1, "Arrays must all be same length."
                for key, value in data.items():
                    self.add_field(field_name=key, fields=value)
            elif isinstance(data, list):
                for ins in data:
                    assert isinstance(ins, Instance), "Must be Instance type, not {}.".format(type(ins))
                    self.append(ins)

            else:
                raise ValueError("data only be dict or list type.")
        self._collater = Collater()


    @property
    def collater(self):
        if self._collater is None:
            self._collater = Collater()
        return self._collater

    @collater.setter
    def collater(self, value):
        assert isinstance(value, Collater)
        self._collater = value

    def __contains__(self, item):
        return item in self.field_arrays

    def __iter__(self):
        def iter_func():
            for idx in range(len(self)):
                yield self[idx]

        return iter_func()

    def _inner_iter(self):
        class Iter_ptr:
            def __init__(self, dataset, idx):
                self.dataset = dataset
                self.idx = idx

            def __getitem__(self, item):
                assert item in self.dataset.field_arrays, "no such field:{} in Instance {}".format(item, self.dataset[
                    self.idx])
                assert self.idx < len(self.dataset.field_arrays[item]), "index:{} out of range".format(self.idx)
                return self.dataset.field_arrays[item][self.idx]

            def __setitem__(self, key, value):
                raise TypeError("You cannot modify value directly.")

            def items(self):
                ins = self.dataset[self.idx]
                return ins.items()

            def __repr__(self):
                return self.dataset[self.idx].__repr__()

        def inner_iter_func():
            for idx in range(len(self)):
                yield Iter_ptr(self, idx)

        return inner_iter_func()

    def __getitem__(self, idx):
        r"""给定int的index，返回一个Instance; 给定slice，返回包含这个slice内容的新的DataSet。

        :param idx: can be int or slice.
        :return: If `idx` is int, return an Instance object.
                If `idx` is slice, return a DataSet object.
        """
        if isinstance(idx, int):
            return Instance(**{name: self.field_arrays[name][idx] for name in self.field_arrays})
        elif isinstance(idx, slice):
            if idx.start is not None and (idx.start >= len(self) or idx.start <= -len(self)):
                raise RuntimeError(f"Start index {idx.start} out of range 0-{len(self) - 1}")
            data_set = DataSet()
            for field_name, field in self.field_arrays.items():
                data_set.add_field(field_name=field_name, fields=field.content[idx], padder=field.padder,
                                   is_input=field.is_input, is_target=field.is_target, ignore_type=field.ignore_type)
            data_set.collater = self.collater.copy_from(self.collater)
            return data_set
        elif isinstance(idx, str):
            if idx not in self:
                raise KeyError("No such field called {} in DataSet.".format(idx))
            return self.field_arrays[idx]
        elif isinstance(idx, list):
            dataset = DataSet()
            for i in idx:
                assert isinstance(i, int), "Only int index allowed."
                instance = self[i]
                dataset.append(instance)
            for field_name, field in self.field_arrays.items():
                dataset.field_arrays[field_name].to(field)
            dataset.collater = self.collater.copy_from(self.collater)
            return dataset
        else:
            raise KeyError("Unrecognized type {} for idx in __getitem__ method".format(type(idx)))

    def __getattr__(self, item):
        # Not tested. Don't use !!
        if item == "field_arrays":
            raise AttributeError
        if isinstance(item, str) and item in self.field_arrays:
            return self.field_arrays[item]

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def __len__(self):
        r"""Fetch the length of the dataset.

        :return length:
        """
        if len(self.field_arrays) == 0:
            return 0
        field = iter(self.field_arrays.values()).__next__()
        return len(field)

    def __repr__(self):
        return str(pretty_table_printer(self))

    def print_field_meta(self):
        r"""
        输出当前field的meta信息, 形似下列的输出::

            +-------------+-------+-------+
            | field_names |   x   |   y   |
            +=============+=======+=======+
            |   is_input  |  True | False |
            |  is_target  | False | False |
            | ignore_type | False |       |
            |  pad_value  |   0   |       |
            +-------------+-------+-------+

        str field_names: DataSet中field的名称
        bool is_input: field是否为input
        bool is_target: field是否为target
        bool ignore_type: 是否忽略该field的type, 一般仅在该field至少为input或target时才有意义
        int pad_value: 该field的pad的值，仅在该field为input或target时有意义
        :return:
        """
        if len(self.field_arrays)>0:
            field_names = ['field_names']
            is_inputs = ['is_input']
            is_targets = ['is_target']
            pad_values = ['pad_value']
            ignore_types = ['ignore_type']

            for name, field_array in self.field_arrays.items():
                field_names.append(name)
                if field_array.is_input:
                    is_inputs.append(True)
                else:
                    is_inputs.append(False)
                if field_array.is_target:
                    is_targets.append(True)
                else:
                    is_targets.append(False)

                if (field_array.is_input or field_array.is_target) and field_array.padder is not None:
                    pad_values.append(field_array.padder.get_pad_val())
                else:
                    pad_values.append(' ')

                if field_array._ignore_type:
                    ignore_types.append(True)
                elif field_array.is_input or field_array.is_target:
                    ignore_types.append(False)
                else:
                    ignore_types.append(' ')
            table = PrettyTable(field_names=field_names)
            fields = [is_inputs, is_targets, ignore_types, pad_values]
            for field in fields:
                table.add_row(field)
            logger.info(table)
            return table


    def append(self, instance):
        r"""
        将一个instance对象append到DataSet后面。

        :param ~fastNLP.Instance instance: 若DataSet不为空，则instance应该拥有和DataSet完全一样的field。

        """
        if len(self.field_arrays) == 0:
            # DataSet has no field yet
            for name, field in instance.fields.items():
                # field = field.tolist() if isinstance(field, np.ndarray) else field
                self.field_arrays[name] = FieldArray(name, [field])  # 第一个样本，必须用list包装起来
        else:
            if len(self.field_arrays) != len(instance.fields):
                raise ValueError(
                    "DataSet object has {} fields, but attempt to append an Instance object with {} fields."
                        .format(len(self.field_arrays), len(instance.fields)))
            for name, field in instance.fields.items():
                assert name in self.field_arrays
                try:
                    self.field_arrays[name].append(field)
                except AppendToTargetOrInputException as e:
                    logger.error(f"Cannot append to field:{name}.")
                    raise e


    def add_fieldarray(self, field_name, fieldarray):
        r"""
        将fieldarray添加到DataSet中.

        :param str field_name: 新加入的field的名称
        :param ~fastNLP.core.FieldArray fieldarray: 需要加入DataSet的field的内容
        :return:
        """
        if not isinstance(fieldarray, FieldArray):
            raise TypeError("Only fastNLP.FieldArray supported.")
        if len(self) != len(fieldarray):
            raise RuntimeError(f"The field to add must have the same size as dataset. "
                               f"Dataset size {len(self)} != field size {len(fieldarray)}")
        fieldarray.name = field_name
        self.field_arrays[field_name] = fieldarray


    def add_field(self, field_name, fields, padder=AutoPadder(), is_input=False, is_target=False, ignore_type=False):
        r"""
        新增一个field
        
        :param str field_name: 新增的field的名称
        :param list fields: 需要新增的field的内容
        :param None,~fastNLP.Padder padder: 如果为None,则不进行pad，默认使用 :class:`~fastNLP.AutoPadder` 自动判断是否需要做pad。
        :param bool is_input: 新加入的field是否是input
        :param bool is_target: 新加入的field是否是target
        :param bool ignore_type: 是否忽略对新加入的field的类型检查
        """

        if len(self.field_arrays) != 0:
            if len(self) != len(fields):
                raise RuntimeError(f"The field to add must have the same size as dataset. "
                                   f"Dataset size {len(self)} != field size {len(fields)}")
        self.field_arrays[field_name] = FieldArray(field_name, fields, is_target=is_target, is_input=is_input,
                                                   padder=padder, ignore_type=ignore_type)


    def delete_instance(self, index):
        r"""
        删除第index个instance

        :param int index: 需要删除的instance的index，序号从0开始。
        """
        assert isinstance(index, int), "Only integer supported."
        if len(self) <= index:
            raise IndexError("{} is too large for as DataSet with {} instances.".format(index, len(self)))
        if len(self) == 1:
            self.field_arrays.clear()
        else:
            for field in self.field_arrays.values():
                field.pop(index)
        return self


    def delete_field(self, field_name):
        r"""
        删除名为field_name的field

        :param str field_name: 需要删除的field的名称.
        """
        self.field_arrays.pop(field_name)
        return self


    def copy_field(self, field_name, new_field_name):
        r"""
        深度copy名为field_name的field到new_field_name

        :param str field_name: 需要copy的field。
        :param str new_field_name: copy生成的field名称
        :return: self
        """
        if not self.has_field(field_name):
            raise KeyError(f"Field:{field_name} not found in DataSet.")
        fieldarray = deepcopy(self.get_field(field_name))
        fieldarray.name = new_field_name
        self.add_fieldarray(field_name=new_field_name, fieldarray=fieldarray)
        return self


    def has_field(self, field_name):
        r"""
        判断DataSet中是否有名为field_name这个field

        :param str field_name: field的名称
        :return bool: 表示是否有名为field_name这个field
        """
        if isinstance(field_name, str):
            return field_name in self.field_arrays
        return False


    def get_field(self, field_name):
        r"""
        获取field_name这个field

        :param str field_name: field的名称
        :return: :class:`~fastNLP.FieldArray`
        """
        if field_name not in self.field_arrays:
            raise KeyError("Field name {} not found in DataSet".format(field_name))
        return self.field_arrays[field_name]


    def get_all_fields(self):
        r"""
        返回一个dict，key为field_name, value为对应的 :class:`~fastNLP.FieldArray`

        :return dict: 返回如上所述的字典
        """
        return self.field_arrays


    def get_field_names(self) -> list:
        r"""
        返回一个list，包含所有 field 的名字

        :return list: 返回如上所述的列表
        """
        return sorted(self.field_arrays.keys())


    def get_length(self):
        r"""
        获取DataSet的元素数量

        :return: int: DataSet中Instance的个数。
        """
        return len(self)


    def rename_field(self, field_name, new_field_name):
        r"""
        将某个field重新命名.

        :param str field_name: 原来的field名称。
        :param str new_field_name: 修改为new_name。
        """
        if field_name in self.field_arrays:
            self.field_arrays[new_field_name] = self.field_arrays.pop(field_name)
            self.field_arrays[new_field_name].name = new_field_name
        else:
            raise KeyError("DataSet has no field named {}.".format(field_name))
        return self


    def set_target(self, *field_names, flag=True, use_1st_ins_infer_dim_type=True):
        r"""
        将field_names的field设置为target

        Example::

            dataset.set_target('labels', 'seq_len')  # 将labels和seq_len这两个field的target属性设置为True
            dataset.set_target('labels', 'seq_lens', flag=False) # 将labels和seq_len的target属性设置为False

        :param str field_names: field的名称
        :param bool flag: 将field_name的target状态设置为flag
        :param bool use_1st_ins_infer_dim_type: 如果为True，将不会check该列是否所有数据都是同样的维度，同样的类型。将直接使用第一
            行的数据进行类型和维度推断本列的数据的类型和维度。
        """
        assert isinstance(flag, bool), "Only bool type supported."
        for name in field_names:
            if name in self.field_arrays:
                try:
                    self.field_arrays[name]._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
                    self.field_arrays[name].is_target = flag
                except SetInputOrTargetException as e:
                    logger.error(f"Cannot set field:{name} as target.")
                    raise e
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self


    def set_input(self, *field_names, flag=True, use_1st_ins_infer_dim_type=True):
        r"""
        将field_names的field设置为input::

            dataset.set_input('words', 'seq_len')   # 将words和seq_len这两个field的input属性设置为True
            dataset.set_input('words', flag=False)  # 将words这个field的input属性设置为False

        :param str field_names: field的名称
        :param bool flag: 将field_name的input状态设置为flag
        :param bool use_1st_ins_infer_dim_type: 如果为True，将不会check该列是否所有数据都是同样的维度，同样的类型。将直接使用第一
            行的数据进行类型和维度推断本列的数据的类型和维度。
        """
        for name in field_names:
            if name in self.field_arrays:
                try:
                    self.field_arrays[name]._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
                    self.field_arrays[name].is_input = flag
                except SetInputOrTargetException as e:
                    logger.error(f"Cannot set field:{name} as input, exception happens at the {e.index} value.")
                    raise e
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self


    def set_ignore_type(self, *field_names, flag=True):
        r"""
        将field设置为忽略类型状态。当某个field被设置了ignore_type, 则在被设置为target或者input时将不进行类型检查，
        默认情况下也不进行pad。如果仍需要pad该field，可通过自定义Padder实现，若该field需要转换为tensor，需要在padder
        中转换，但不需要在padder中移动到gpu。

        :param str field_names: field的名称
        :param bool flag: 将field_name的ignore_type状态设置为flag
        :return:
        """
        assert isinstance(flag, bool), "Only bool type supported."
        for name in field_names:
            if name in self.field_arrays:
                self.field_arrays[name].ignore_type = flag
            else:
                raise KeyError("{} is not a valid field name.".format(name))
        return self


    def set_padder(self, field_name, padder):
        r"""
        为field_name设置padder::

            from fastNLP import EngChar2DPadder
            padder = EngChar2DPadder()
            dataset.set_padder('chars', padder)  # 则chars这个field会使用EngChar2DPadder进行pad操作

        :param str field_name: 设置field的padding方式为padder
        :param None,~fastNLP.Padder padder: 设置为None即删除padder, 即对该field不进行pad操作。
        """
        if field_name not in self.field_arrays:
            raise KeyError("There is no field named {}.".format(field_name))
        self.field_arrays[field_name].set_padder(padder)
        return self


    def set_pad_val(self, field_name, pad_val):
        r"""
        为某个field设置对应的pad_val.

        :param str field_name: 修改该field的pad_val
        :param int pad_val: 该field的padder会以pad_val作为padding index
        """
        if field_name not in self.field_arrays:
            raise KeyError("There is no field named {}.".format(field_name))
        self.field_arrays[field_name].set_pad_val(pad_val)
        return self


    def get_input_name(self):
        r"""
        返回所有is_input被设置为True的field名称

        :return list: 里面的元素为被设置为input的field名称
        """
        return [name for name, field in self.field_arrays.items() if field.is_input]


    def get_target_name(self):
        r"""
        返回所有is_target被设置为True的field名称

        :return list: 里面的元素为被设置为target的field名称
        """
        return [name for name, field in self.field_arrays.items() if field.is_target]


    def apply_field(self, func, field_name, new_field_name=None, **kwargs):
        r"""
        将DataSet中的每个instance中的名为 `field_name` 的field传给func，并获取它的返回值。

        :param callable func: input是instance中名为 `field_name` 的field的内容。
        :param str field_name: 传入func的是哪个field。
        :param None,str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
            盖之前的field。如果为None则不创建新的field。
        :param optional kwargs: 支持输入is_input,is_target,ignore_type

            1. is_input: bool, 如果为True则将名为 `new_field_name` 的field设置为input

            2. is_target: bool, 如果为True则将名为 `new_field_name` 的field设置为target

            3. ignore_type: bool, 如果为True则将名为 `new_field_name` 的field的ignore_type设置为true, 忽略其类型

            4. use_tqdm: bool, 是否使用tqdm显示预处理进度

            5. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称

        :return List[Any]:   里面的元素为func的返回值，所以list长度为DataSet的长度
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if not self.has_field(field_name=field_name):
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        return self.apply(func, new_field_name, _apply_field=field_name, **kwargs)


    def apply_field_more(self, func, field_name, modify_fields=True, **kwargs):
        r"""
        将 ``DataSet`` 中的每个 ``Instance`` 中的名为 `field_name` 的field 传给 func，并获取它的返回值。
        func 可以返回一个或多个 field 上的结果。
        
        .. note::
            ``apply_field_more`` 与 ``apply_field`` 的区别参考 :meth:`~fastNLP.DataSet.apply_more` 中关于 ``apply_more`` 与
            ``apply`` 区别的介绍。
            
        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param str field_name: 传入func的是哪个field。
        :param bool modify_fields: 是否用结果修改 `DataSet` 中的 `Field`， 默认为 True
        :param optional kwargs: 支持输入is_input,is_target,ignore_type

            1. is_input: bool, 如果为True则将被修改的field设置为input

            2. is_target: bool, 如果为True则将被修改的field设置为target

            3. ignore_type: bool, 如果为True则将被修改的field的ignore_type设置为true, 忽略其类型

            4. use_tqdm: bool, 是否使用tqdm显示预处理进度

            5. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称

        :return Dict[str:Field]: 返回一个字典
        """
        assert len(self) != 0, "Null DataSet cannot use apply_field()."
        if not self.has_field(field_name=field_name):
            raise KeyError("DataSet has no field named `{}`.".format(field_name))
        return self.apply_more(func, modify_fields, _apply_field=field_name, **kwargs)

    
    def _add_apply_field(self, results, new_field_name, kwargs):
        r"""
        将results作为加入到新的field中，field名称为new_field_name

        :param List[str] results: 一般是apply*()之后的结果
        :param str new_field_name: 新加入的field的名称
        :param dict kwargs: 用户apply*()时传入的自定义参数
        :return:
        """
        extra_param = {}
        if 'is_input' in kwargs:
            extra_param['is_input'] = kwargs['is_input']
        if 'is_target' in kwargs:
            extra_param['is_target'] = kwargs['is_target']
        if 'ignore_type' in kwargs:
            extra_param['ignore_type'] = kwargs['ignore_type']
        if new_field_name in self.field_arrays:
            # overwrite the field, keep same attributes
            old_field = self.field_arrays[new_field_name]
            if 'is_input' not in extra_param:
                extra_param['is_input'] = old_field.is_input
            if 'is_target' not in extra_param:
                extra_param['is_target'] = old_field.is_target
            if 'ignore_type' not in extra_param:
                extra_param['ignore_type'] = old_field.ignore_type
            self.add_field(field_name=new_field_name, fields=results, is_input=extra_param["is_input"],
                           is_target=extra_param["is_target"], ignore_type=extra_param['ignore_type'],
                           padder=self.get_field(new_field_name).padder)
        else:
            self.add_field(field_name=new_field_name, fields=results, is_input=extra_param.get("is_input", None),
                           is_target=extra_param.get("is_target", None),
                           ignore_type=extra_param.get("ignore_type", False))

    def apply_more(self, func, modify_fields=True, **kwargs):
        r"""
        将 ``DataSet`` 中每个 ``Instance`` 传入到func中，并获取它的返回值。func可以返回一个或多个 field 上的结果。
        
        .. note::
            ``apply_more`` 与 ``apply`` 的区别：
            
            1. ``apply_more`` 可以返回多个 field 的结果， ``apply`` 只可以返回一个field 的结果；
            
            2. ``apply_more`` 的返回值是一个字典，每个 key-value 对中的 key 表示 field 的名字，value 表示计算结果；
            
            3. ``apply_more`` 默认修改 ``DataSet`` 中的 field ，``apply`` 默认不修改。

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance`` ，返回值是一个字典，key 是field 的名字，value 是对应的结果
        :param bool modify_fields: 是否用结果修改 ``DataSet`` 中的 ``Field`` ， 默认为 True
        :param optional kwargs: 支持输入is_input,is_target,ignore_type

            1. is_input: bool, 如果为True则将被修改的的field设置为input

            2. is_target: bool, 如果为True则将被修改的的field设置为target

            3. ignore_type: bool, 如果为True则将被修改的的field的ignore_type设置为true, 忽略其类型

            4. use_tqdm: bool, 是否使用tqdm显示预处理进度

            5. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称

        :return Dict[str:Field]: 返回一个字典
        """
        # 返回 dict , 检查是否一直相同
        assert callable(func), "The func you provide is not callable."
        assert len(self) != 0, "Null DataSet cannot use apply()."
        idx = -1
        try:
            results = {}
            for idx, ins in tqdm(enumerate(self._inner_iter()), total=len(self), dynamic_ncols=True,
                                 desc=kwargs.get('tqdm_desc', ''),
                                 leave=False, disable=not kwargs.get('use_tqdm', False)):
                if "_apply_field" in kwargs:
                    res = func(ins[kwargs["_apply_field"]])
                else:
                    res = func(ins)
                if not isinstance(res, dict):
                    raise ApplyResultException("The result of func is not a dict", idx)
                if idx == 0:
                    for key, value in res.items():
                        results[key] = [value]
                else:
                    for key, value in res.items():
                        if key not in results:
                            raise ApplyResultException("apply results have different fields", idx)
                        results[key].append(value)
                    if len(res) != len(results):
                        raise ApplyResultException("apply results have different fields", idx)
        except Exception as e:
            if idx != -1:
                if isinstance(e, ApplyResultException):
                    logger.error(e.msg)
                logger.error("Exception happens at the `{}`th instance.".format(idx))
            raise e
    
        if modify_fields is True:
            for field, result in results.items():
                self._add_apply_field(result, field, kwargs)
    
        return results


    def apply(self, func, new_field_name=None, **kwargs):
        r"""
        将DataSet中每个instance传入到func中，并获取它的返回值.

        :param callable func: 参数是 ``DataSet`` 中的 ``Instance``
        :param None,str new_field_name: 将func返回的内容放入到 `new_field_name` 这个field中，如果名称与已有的field相同，则覆
            盖之前的field。如果为None则不创建新的field。
        :param optional kwargs: 支持输入is_input,is_target,ignore_type

            1. is_input: bool, 如果为True则将 `new_field_name` 的field设置为input

            2. is_target: bool, 如果为True则将 `new_field_name` 的field设置为target

            3. ignore_type: bool, 如果为True则将 `new_field_name` 的field的ignore_type设置为true, 忽略其类型

            4. use_tqdm: bool, 是否使用tqdm显示预处理进度

            5. tqdm_desc: str, 当use_tqdm为True时，可以显示当前tqdm正在处理的名称
            
        :return List[Any]: 里面的元素为func的返回值，所以list长度为DataSet的长度
        """
        assert callable(func), "The func you provide is not callable."
        assert len(self) != 0, "Null DataSet cannot use apply()."
        idx = -1
        try:
            results = []
            for idx, ins in tqdm(enumerate(self._inner_iter()), total=len(self), dynamic_ncols=True, leave=False,
                                 desc=kwargs.get('tqdm_desc', ''),
                                 disable=not kwargs.get('use_tqdm', False)):
                if "_apply_field" in kwargs:
                    results.append(func(ins[kwargs["_apply_field"]]))
                else:
                    results.append(func(ins))
        except BaseException as e:
            if idx != -1:
                logger.error("Exception happens at the `{}`th instance.".format(idx))
            raise e

        if new_field_name is not None:
            self._add_apply_field(results, new_field_name, kwargs)

        return results


    def add_seq_len(self, field_name: str, new_field_name=Const.INPUT_LEN):
        r"""
        将使用len()直接对field_name中每个元素作用，将其结果作为sequence length, 并放入seq_len这个field。

        :param field_name: str.
        :param new_field_name: str. 新的field_name
        :return:
        """
        if self.has_field(field_name=field_name):
            self.apply_field(len, field_name, new_field_name=new_field_name)
        else:
            raise KeyError(f"Field:{field_name} not found.")
        return self


    def drop(self, func, inplace=True):
        r"""
        func接受一个Instance，返回bool值。返回值为True时，该Instance会被移除或者不会包含在返回的DataSet中。

        :param callable func: 接受一个Instance作为参数，返回bool值。为True时删除该instance
        :param bool inplace: 是否在当前DataSet中直接删除instance；如果为False，将返回一个新的DataSet。

        :return: DataSet
        """
        if inplace:
            results = [ins for ins in self._inner_iter() if not func(ins)]
            for name, old_field in self.field_arrays.items():
                self.field_arrays[name].content = [ins[name] for ins in results]
            return self
        else:
            results = [ins for ins in self if not func(ins)]
            if len(results) != 0:
                dataset = DataSet(results)
                for field_name, field in self.field_arrays.items():
                    dataset.field_arrays[field_name].to(field)
                return dataset
            else:
                return DataSet()


    def split(self, ratio, shuffle=True):
        r"""
        将DataSet按照ratio的比例拆分，返回两个DataSet

        :param float ratio: 0<ratio<1, 返回的第一个DataSet拥有 `(1-ratio)` 这么多数据，第二个DataSet拥有`ratio`这么多数据
        :param bool shuffle: 在split前是否shuffle一下
        :return: [ :class:`~fastNLP.读取后的DataSet` , :class:`~fastNLP.读取后的DataSet` ]
        """
        assert len(self) > 1, f'DataSet with {len(self)} instance cannot be split.'
        assert isinstance(ratio, float)
        assert 0 < ratio < 1
        all_indices = [_ for _ in range(len(self))]
        if shuffle:
            np.random.shuffle(all_indices)
        split = int(ratio * len(self))
        if split == 0:
            error_msg = f'Dev DataSet has {split} instance after split.'
            logger.error(error_msg)
            raise IndexError(error_msg)
        dev_indices = all_indices[:split]
        train_indices = all_indices[split:]
        dev_set = DataSet()
        train_set = DataSet()
        for idx in dev_indices:
            dev_set.append(self[idx])
        for idx in train_indices:
            train_set.append(self[idx])
        for field_name in self.field_arrays:
            train_set.field_arrays[field_name].to(self.field_arrays[field_name])
            dev_set.field_arrays[field_name].to(self.field_arrays[field_name])

        train_set.collater.copy_from(self.collater)
        dev_set.collater.copy_from(self.collater)
        return train_set, dev_set


    def save(self, path):
        r"""
        保存DataSet.

        :param str path: 将DataSet存在哪个路径
        """
        with open(path, 'wb') as f:
            _pickle.dump(self, f)


    @staticmethod
    def load(path):
        r"""
        从保存的DataSet pickle文件的路径中读取DataSet

        :param str path: 从哪里读取DataSet
        :return: 读取后的 :class:`~fastNLP.读取后的DataSet`。
        """
        with open(path, 'rb') as f:
            d = _pickle.load(f)
            assert isinstance(d, DataSet), "The object is not DataSet, but {}.".format(type(d))
        return d


    def add_collate_fn(self, fn, name=None):
        r"""
        添加 CollateFn，collate_fn允许在生成的batch的过程中动态生成一些数据(在DataSetIter作为迭代器的情况下有效，默认情况下就是用的
        这个)。支持依次添加多个collate_fn, 如果相同的key，后面的collate_fn的结果覆盖前面的collate_fn的结果。

        :param callable fn: 传入一个可调用的function, 该function可接受的参数为List[(ind1, instance1), (ind2, instance2)]
            (某个batch被选中的所有的indice以及instance),其中ind1/ind2是该instance在dataset中的index，instance1/instance2是
            这次batch取出来的数据，包含了所有的field。返回值需要为两个dict，第一个dict的值将被认为是input，第二个dict的值被认为是
            target，返回的值至多允许一个空dict。若返回的dict中包含了被设置为input或target的field的名称，将覆盖dataset中的field。
            fastNLP不会将collate_fn的返回结果pad和转换为tensor，需要在collate_fn中完成pad和转换为tensor（不需要将tensor移动到
            gpu中，fastNLP会自动将其移动到特定gpu）。不要修改传入collate_fn中的数据，否则可能导致未知问题。
        :param str,int name: collate_fn的名称，如果不传入，默认使用自增长的数字作为key。相同的name会覆盖之前的collate_fn。
        """
        assert callable(fn), "You must pass in a callable object."
        self.collater.add_fn(fn, name=name)


    def delete_collate_fn(self, name=None):
        r"""
        删除某个collate_fn

        :param str,int name: 如果为None，则删除最近加入的collate_fn
        :return:
        """
        self.collater.delete_fn(name)


    def _collate_batch(self, ins_list):
        return self.collater.collate_batch(ins_list)

    def concat(self, dataset, inplace=True, field_mapping=None):
        """
        将当前dataset与输入的dataset结合成一个更大的dataset，需要保证两个dataset都包含了相同的field。结合后的dataset的input,target
            以及collate_fn以当前dataset为准。当dataset中包含的field多于当前的dataset，则多余的field会被忽略；若dataset中未包含所有
            当前dataset含有field，则会报错。

        :param DataSet, dataset: 需要和当前dataset concat的dataset
        :param bool, inplace: 是否直接将dataset组合到当前dataset中
        :param dict, field_mapping: 当dataset中的field名称和当前dataset不一致时，需要通过field_mapping把输入的dataset中的field
            名称映射到当前field. field_mapping为dict类型，key为dataset中的field名称，value是需要映射成的名称

        :return: DataSet
        """
        assert isinstance(dataset, DataSet), "Can only concat two datasets."

        fns_in_this_dataset = set(self.get_field_names())
        fns_in_other_dataset = dataset.get_field_names()
        reverse_field_mapping = {}
        if field_mapping is not None:
            fns_in_other_dataset = [field_mapping.get(fn, fn) for fn in fns_in_other_dataset]
            reverse_field_mapping = {v:k for k, v in field_mapping.items()}
        fns_in_other_dataset = set(fns_in_other_dataset)
        fn_not_seen = list(fns_in_this_dataset - fns_in_other_dataset)

        if fn_not_seen:
            raise RuntimeError(f"The following fields are not provided in the dataset:{fn_not_seen}")

        if inplace:
            ds = self
        else:
            ds = deepcopy(self)

        for fn in fns_in_this_dataset:
            ds.get_field(fn).content.extend(deepcopy(dataset.get_field(reverse_field_mapping.get(fn, fn)).content))

        return ds

#@title from fastNLP.core import LossBase, MetricBase
class LossBase(object):
    r"""
    所有loss的基类。如果需要结合到Trainer之中需要实现get_loss方法
    """
    
    def __init__(self):
        self._param_map = {}  # key是fun的参数，value是以该值从传入的dict取出value
        self._checked = False

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    def get_loss(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return: torch.Tensor
        """
        raise NotImplementedError

    
    def _init_param_map(self, key_map=None, **kwargs):
        r"""检查key_map和其他参数map，并将这些映射关系添加到self._param_map

        :param dict key_map: 表示key的映射关系
        :param kwargs: key word args里面的每一个的键-值对都会被构造成映射关系
        :return: None
        """
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError("key_map must be `dict`, got {}.".format(type(key_map)))
            for key, value in key_map.items():
                if value is None:
                    self._param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self._param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")
        
        # check consistence between signature and _param_map
        func_spect = inspect.getfullargspec(self.get_loss)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.get_loss)}. Please check the "
                    f"initialization parameters, or change its signature.")
        
        # evaluate should not have varargs.
        # if func_spect.varargs:
        #     raise NameError(f"Delete `*{func_spect.varargs}` in {get_func_signature(self.get_loss)}(Do not use "
        #                     f"positional argument.).")

    def __call__(self, pred_dict, target_dict, check=False):
        r"""
        :param dict pred_dict: 模型的forward函数返回的dict
        :param dict target_dict: DataSet.batch_y里的键-值对所组成的dict
        :param Boolean check: 每一次执行映射函数的时候是否检查映射表，默认为不检查
        :return:
        """

        if not self._checked:
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.get_loss)}.")
            
            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]
        
        # missing
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in target_dict:
                    duplicated.append(input_arg)
            check_res = _check_arg_dict_list(self.get_loss, [mapped_pred_dict, mapped_target_dict])
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self._param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                    f"in `{self.__class__.__name__}`)"
            
            check_res = _CheckRes(missing=replaced_missing,
                                  unused=check_res.unused,
                                  duplicated=duplicated,
                                  required=check_res.required,
                                  all_needed=check_res.all_needed,
                                  varargs=check_res.varargs)
            
            if check_res.missing or check_res.duplicated:
                raise _CheckError(check_res=check_res,
                                  func_signature=_get_func_signature(self.get_loss))
            self._checked = True

        refined_args = _build_args(self.get_loss, **mapped_pred_dict, **mapped_target_dict)
        
        loss = self.get_loss(**refined_args)
        self._checked = True
        
        return loss
    

class MetricBase(object):

    def __init__(self):
        self._param_map = {}  # key is param in function, value is input param.
        self._checked = False
        self._metric_name = self.__class__.__name__

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset=True):
        raise NotImplemented

    def set_metric_name(self, name: str):
        r"""
        设置metric的名称，默认是Metric的class name.

        :param str name:
        :return: self
        """
        self._metric_name = name
        return self


    def get_metric_name(self):
        r"""
        返回metric的名称
        
        :return:
        """
        return self._metric_name


    def _init_param_map(self, key_map=None, **kwargs):
        r"""检查key_map和其他参数map，并将这些映射关系添加到self._param_map

        :param dict key_map: 表示key的映射关系
        :param kwargs: key word args里面的每一个的键-值对都会被构造成映射关系
        :return: None
        """
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError("key_map must be `dict`, got {}.".format(type(key_map)))
            for key, value in key_map.items():
                if value is None:
                    self._param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self._param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")

        # check consistence between signature and _param_map
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

    def __call__(self, pred_dict, target_dict):
        r"""
        这个方法会调用self.evaluate 方法.
        在调用之前，会进行以下检测:
            1. self.evaluate当中是否有varargs, 这是不支持的.
            2. self.evaluate当中所需要的参数是否既不在``pred_dict``也不在``target_dict``.
            3. self.evaluate当中所需要的参数是否既在``pred_dict``也在``target_dict``.

            除此以外，在参数被传入self.evaluate以前，这个函数会检测``pred_dict``和``target_dict``当中没有被用到的参数
            如果kwargs是self.evaluate的参数，则不会检测
        :param pred_dict: 模型的forward函数或者predict函数返回的dict
        :param target_dict: DataSet.batch_y里的键-值对所组成的dict(即is_target=True的fields的内容)
        :return:
        """

        if not self._checked:
            if not callable(self.evaluate):
                raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.evaluate)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]

        # missing
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in target_dict:
                    duplicated.append(input_arg)
            check_res = _check_arg_dict_list(self.evaluate, [mapped_pred_dict, mapped_target_dict])
            # only check missing.
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self._param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                                                                         f"in `{self.__class__.__name__}`)"

            check_res = _CheckRes(missing=replaced_missing,
                                  unused=check_res.unused,
                                  duplicated=duplicated,
                                  required=check_res.required,
                                  all_needed=check_res.all_needed,
                                  varargs=check_res.varargs)

            if check_res.missing or check_res.duplicated:
                raise _CheckError(check_res=check_res,
                                  func_signature=_get_func_signature(self.evaluate))
            self._checked = True
        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)

        return

#@title losses & metrics
def _prepare_metrics(metrics):
    r"""

    Prepare list of Metric based on input
    :param metrics:
    :return: List[fastNLP.MetricBase]
    """
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                if isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
                    if not callable(metric.get_metric):
                        raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
                    _metrics.append(metric)
                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `fastNLP.MetricBase`, not `{type(metric)}`.")
        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(f"The type of metrics should be `list[fastNLP.MetricBase]` or `fastNLP.MetricBase`, "
                            f"got {type(metrics)}.")
    return _metrics


class LossInForward(LossBase):
    def __init__(self, loss_key='loss'):
        super().__init__()
        if not isinstance(loss_key, str):
            raise TypeError(f"Only str allowed for loss_key, got {type(loss_key)}.")
        self.loss_key = loss_key

    def get_loss(self, **kwargs):
        if self.loss_key not in kwargs:
            check_res = CheckRes(missing=[self.loss_key + f"(assign to `{self.loss_key}` " \
                                                                        f"in `{self.__class__.__name__}`"],
                                 unused=[],
                                 duplicated=[],
                                 required=[],
                                 all_needed=[],
                                 varargs=[])
            raise CheckError(check_res=check_res, func_signature=get_func_signature(self.get_loss))
        return kwargs[self.loss_key]

    def __call__(self, pred_dict, target_dict, check=False):

        loss = self.get_loss(**pred_dict)

        if not (isinstance(loss, torch.Tensor) and len(loss.size()) == 0):
            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"Loss excepted to be a torch.Tensor, got {type(loss)}")
            raise RuntimeError(f"The size of loss excepts to be torch.Size([]), got {loss.size()}")

        return loss

def _prepare_losser(losser):
    if losser is None:
        losser = LossInForward()
        return losser
    elif isinstance(losser, LossBase):
        return losser
    else:
        raise TypeError(f"Type of loss should be `fastNLP.LossBase`, got {type(losser)}")