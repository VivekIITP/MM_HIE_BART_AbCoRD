import torch
import threading
import contextlib
import inspect
import numpy as np
from prettytable import PrettyTable
from typing import Union, List, Dict, Any, Optional, Tuple
from pkg_resources import parse_version
from collections import defaultdict, Counter, namedtuple
from torch.nn.parallel.parallel_apply import get_a_var
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
from fastNLP import logger

class _pseudo_tqdm:

    def __init__(self, **kwargs):
        self.logger = logger

    def write(self, info):
        self.logger.info(info)

    def set_postfix_str(self, info):
        self.logger.info(info)

    def __getattr__(self, item):
        def pass_func(*args, **kwargs):
            pass

        return pass_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

CheckRes = namedtuple('CheckRes', ['missing', 'unused', 'duplicated', 'required', 'all_needed', 'varargs'])#, verbose=False)
_CheckRes = namedtuple('_CheckRes', ['missing', 'unused', 'duplicated', 'required', 'all_needed','varargs'])

class CheckError(Exception):
    """

    CheckError. Used in losses.LossBase, metrics.MetricBase.
    """

    def __init__(self, check_res: CheckRes, func_signature: str):
        errs = [f'Problems occurred when calling `{func_signature}`']

        if check_res.varargs:
            errs.append(f"\tvarargs: {check_res.varargs}(Does not support pass positional arguments, please delete it)")
        if check_res.missing:
            errs.append(f"\tmissing param: {check_res.missing}")
        if check_res.duplicated:
            errs.append(f"\tduplicated param: {check_res.duplicated}")
        if check_res.unused:
            errs.append(f"\tunused param: {check_res.unused}")

        Exception.__init__(self, '\n'.join(errs))

        self.check_res = check_res
        self.func_signature = func_signature

class _CheckError(Exception):
    r"""

    _CheckError. Used in losses.LossBase, metrics.MetricBase.
    """

    def __init__(self, check_res: _CheckRes, func_signature: str):
        errs = [f'Problems occurred when calling `{func_signature}`']

        if check_res.varargs:
            errs.append(f"\tvarargs: {check_res.varargs}(Does not support pass positional arguments, please delete it)")
        if check_res.missing:
            errs.append(f"\tmissing param: {check_res.missing}")
        if check_res.duplicated:
            errs.append(f"\tduplicated param: {check_res.duplicated}")
        if check_res.unused:
            errs.append(f"\tunused param: {check_res.unused}")

        Exception.__init__(self, '\n'.join(errs))

        self.check_res = check_res
        self.func_signature = func_signature


def _get_model_device(model):
    r"""
    传入一个nn.Module的模型，获取它所在的device

    :param model: nn.Module
    :return: torch.device,None 如果返回值为None，说明这个模型没有任何参数。
    """
    # TODO 这个函数存在一定的风险，因为同一个模型可能存在某些parameter不在显卡中，比如BertEmbedding. 或者跨显卡
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


def _build_args(func, **kwargs):
    spect = inspect.getfullargspec(func)
    if spect.varkw is not None:
        return kwargs
    needed_args = set(spect.args)
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output

def _check_arg_dict_list(func, args):
    if isinstance(args, dict):
        arg_dict_list = [args]
    else:
        arg_dict_list = args
    assert callable(func) and isinstance(arg_dict_list, (list, tuple))
    assert len(arg_dict_list) > 0 and isinstance(arg_dict_list[0], dict)
    spect = inspect.getfullargspec(func)
    all_args = set([arg for arg in spect.args if arg != 'self'])
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    default_args = set(spect.args[start_idx:])
    require_args = all_args - default_args
    input_arg_count = Counter()
    for arg_dict in arg_dict_list:
        input_arg_count.update(arg_dict.keys())
    duplicated = [name for name, val in input_arg_count.items() if val > 1]
    input_args = set(input_arg_count.keys())
    missing = list(require_args - input_args)
    unused = list(input_args - all_args)
    varargs = [] if not spect.varargs else [spect.varargs]
    return CheckRes(missing=missing,
                    unused=unused,
                    duplicated=duplicated,
                    required=list(require_args),
                    all_needed=list(all_args),
                    varargs=varargs)
    

def get_func_signature(func):
    """

    Given a function or method, return its signature.
    For example:
    (1) function
        def func(a, b='a', *args):
            xxxx
        get_func_signature(func) # 'func(a, b='a', *args)'
    (2) method
        class Demo:
            def __init__(self):
                xxx
            def forward(self, a, b='a', **args)
        demo = Demo()
        get_func_signature(demo.forward) # 'Demo.forward(self, a, b='a', **args)'
    :param func: a function or a method
    :return: str or None
    """
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        signature = inspect.signature(func)
        signature_str = str(signature)
        if len(signature_str) > 2:
            _self = '(self, '
        else:
            _self = '(self'
        signature_str = class_name + '.' + func.__name__ + _self + signature_str[1:]
        return signature_str
    elif inspect.isfunction(func):
        signature = inspect.signature(func)
        signature_str = str(signature)
        signature_str = func.__name__ + signature_str
        return signature_str


def _get_func_signature(func):
    r"""

    Given a function or method, return its signature.
    For example:
    
    1 function::
    
        def func(a, b='a', *args):
            xxxx
        get_func_signature(func) # 'func(a, b='a', *args)'
        
    2 method::
    
        class Demo:
            def __init__(self):
                xxx
            def forward(self, a, b='a', **args)
        demo = Demo()
        get_func_signature(demo.forward) # 'Demo.forward(self, a, b='a', **args)'
        
    :param func: a function or a method
    :return: str or None
    """
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        signature = inspect.signature(func)
        signature_str = str(signature)
        if len(signature_str) > 2:
            _self = '(self, '
        else:
            _self = '(self'
        signature_str = class_name + '.' + func.__name__ + _self + signature_str[1:]
        return signature_str
    elif inspect.isfunction(func):
        signature = inspect.signature(func)
        signature_str = str(signature)
        signature_str = func.__name__ + signature_str
        return signature_str


def _is_function_or_method(func):
    """

    :param func:
    :return:
    """
    if not inspect.ismethod(func) and not inspect.isfunction(func):
        return False
    return True

def _check_function_or_method(func):
    if not _is_function_or_method(func):
        raise TypeError(f"{type(func)} is not a method or function.")

IGNORE_CHECK_LEVEL = 0
WARNING_CHECK_LEVEL = 1
STRICT_CHECK_LEVEL = 2


def _check_loss_evaluate(prev_func_signature: str, func_signature: str, check_res: CheckRes, 
                         pred_dict: dict, target_dict: dict, dataset, check_level=0):
    errs = []
    unuseds = []
    _unused_field = []
    _unused_param = []
    suggestions = []
    # if check_res.varargs:
    #     errs.append(f"\tvarargs: *{check_res.varargs}")
    #     suggestions.append(f"Does not support pass positional arguments, please delete *{check_res.varargs}.")

    if check_res.unused:
        for _unused in check_res.unused:
            if _unused in target_dict:
                _unused_field.append(_unused)
            else:
                _unused_param.append(_unused)
        if _unused_field:
            unuseds.append(f"\tunused field: {_unused_field}")
        if _unused_param:
            unuseds.append(f"\tunused param: {_unused_param}")  # output from predict or forward

    module_name = func_signature.split('.')[0]
    if check_res.missing:
        errs.append(f"\tmissing param: {check_res.missing}")
        import re
        mapped_missing = []  # 提供了映射的参数
        unmapped_missing = []  # 没有指定映射的参数
        input_func_map = {}
        for _miss_ in check_res.missing:
            # they shoudl like 'SomeParam(assign to xxx)'
            _miss = _miss_.split('(')[0]
            matches = re.findall("(?<=`)[a-zA-Z0-9]*?(?=`)", _miss_)
            if len(matches) == 2:
                fun_arg, module_name = matches
                input_func_map[_miss] = fun_arg
                if fun_arg == _miss:
                    unmapped_missing.append(_miss)
                else:
                    mapped_missing.append(_miss)
            else:
                unmapped_missing.append(_miss)

        for _miss in mapped_missing + unmapped_missing:
            if _miss in dataset:
                suggestions.append(f"Set `{_miss}` as target.")
            else:
                _tmp = ''
                if check_res.unused:
                    _tmp = f"Check key assignment for `{input_func_map.get(_miss,_miss)}` when initialize {module_name}."
                if _tmp:
                    _tmp += f' Or provide `{_miss}` in DataSet or the output of {prev_func_signature}. '
                else:
                    _tmp = f'Provide `{_miss}` in DataSet or the output of {prev_func_signature}.'
                if not dataset.collater.is_empty():
                    _tmp += f'Or you need to add `{_miss}` in the output of your collate_fn. '
                suggestions.append(_tmp)

    if check_res.duplicated:
        errs.append(f"\tduplicated param: {check_res.duplicated}.")
        suggestions.append(f"Delete {check_res.duplicated} in the output of "
                           f"{prev_func_signature} or do not set {check_res.duplicated} as targets. ")

    if len(errs) > 0:
        errs.extend(unuseds)
    elif check_level == STRICT_CHECK_LEVEL:
        errs.extend(unuseds)

    if len(errs) > 0:
        errs.insert(0, f'Problems occurred when calling {func_signature}')
        sugg_str = ""
        if len(suggestions) > 1:
            for idx, sugg in enumerate(suggestions):
                if idx > 0:
                    sugg_str += '\t\t\t'
                sugg_str += f'({idx + 1}). {sugg}\n'
            sugg_str = sugg_str[:-1]
        else:
            sugg_str += suggestions[0]
        errs.append(f'\ttarget field: {list(target_dict.keys())}')
        errs.append(f'\tparam from {prev_func_signature}: {list(pred_dict.keys())}')
        err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        raise NameError(err_str)
    if check_res.unused:
        if check_level == WARNING_CHECK_LEVEL:
            if not module_name:
                module_name = func_signature.split('.')[0]
            _unused_warn = f'{check_res.unused} is not used by {module_name}.'
            warnings.warn(message=_unused_warn)


def _move_model_to_device(model, device):
    r"""
    将model移动到device

    :param model: torch.nn.DataParallel or torch.nn.Module. 当为torch.nn.DataParallel, 则只是调用一次cuda。device必须为
        None。
    :param str,int,torch.device,list(int),list(torch.device) device: 将模型load到哪个设备。默认为None，即Trainer不对模型
        的计算位置进行管理。支持以下的输入:

        1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中, 可见的第一个GPU中,
        可见的第二个GPU中;

        2. torch.device：将模型装载到torch.device上。

        3. int: 将使用device_id为该值的gpu进行训练

        4. list(int)：如果多于1个device，将使用torch.nn.DataParallel包裹model, 并使用传入的device。

        5. None. 为None则不对模型进行任何处理，如果传入的model为torch.nn.DataParallel该值必须为None。

    :return: torch.nn.DataParallel or torch.nn.Module
    """
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     raise RuntimeError("model of `torch.nn.parallel.DistributedDataParallel` is not supported right now.")

    if device is None:
        if isinstance(model, torch.nn.DataParallel):
            model.cuda(model.device_ids[0])
        return model
    else:
        if not torch.cuda.is_available() and ((isinstance(device, str) and device!='cpu') or
         (isinstance(device, torch.device) and device.type != 'cpu')):
            raise ValueError("There is no usable gpu. set `device` as `cpu` or `None`.")

    if isinstance(model, torch.nn.DataParallel):
        raise RuntimeError("When model is `torch.nn.DataParallel`, the device has to be `None`.")

    if isinstance(device, int):
        assert device > -1, "device can only be non-negative integer"
        assert torch.cuda.device_count() > device, "Only has {} gpus, cannot use device {}.".format(
            torch.cuda.device_count(),
            device)
        device = torch.device('cuda:{}'.format(device))
    elif isinstance(device, str):
        device = torch.device(device)
        if device.type == 'cuda' and device.index is not None:
            assert device.index < torch.cuda.device_count(), "Only has {} gpus, cannot use device cuda:{}.".format(
                torch.cuda.device_count(),
                device)
    elif isinstance(device, torch.device):
        if device.type == 'cuda' and device.index is not None:
            assert device.index < torch.cuda.device_count(), "Only has {} gpus, cannot use device cuda:{}.".format(
                torch.cuda.device_count(),
                device)
    elif isinstance(device, list):
        types = set([type(d) for d in device])
        assert len(types) == 1, "Mixed type in device, only `int` allowed."
        assert list(types)[0] == int, "Only int supported for multiple devices."
        assert len(set(device)) == len(device), "Duplicated device id found in device."
        for d in device:
            assert d > -1, "Only non-negative device id allowed."
        if len(device) > 1:
            output_device = device[0]
            model = nn.DataParallel(model, device_ids=device, output_device=output_device)
        device = torch.device(device[0])
    else:
        raise TypeError("Unsupported device type.")
    model = model.to(device)
    return model


def _move_dict_value_to_device(*args, device: torch.device, non_blocking=False):
    r"""

    move data to model's device, element in *args should be dict. This is a inplace change.
    :param device: torch.device
    :param non_blocking: bool, 是否异步将数据转移到cpu, 需要tensor使用pin_memory()
    :param args:
    :return:
    """
    if not torch.cuda.is_available() or device is None:
        return

    if not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device`, got `{type(device)}`")

    for arg in args:
        if isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, torch.Tensor):
                    arg[key] = value.to(device, non_blocking=non_blocking)
        else:
            raise TypeError("Only support `dict` type right now.")


def _check_forward_error(forward_func, batch_x, dataset, check_level):
    check_res = _check_arg_dict_list(forward_func, batch_x)
    func_signature = _get_func_signature(forward_func)

    errs = []
    suggestions = []
    _unused = []

    # if check_res.varargs:
    #     errs.append(f"\tvarargs: {check_res.varargs}")
    #     suggestions.append(f"Does not support pass positional arguments, please delete *{check_res.varargs}.")
    if check_res.missing:
        errs.append(f"\tmissing param: {check_res.missing}")
        _miss_in_dataset = []
        _miss_out_dataset = []
        for _miss in check_res.missing:
            if _miss in dataset:
                _miss_in_dataset.append(_miss)
            else:
                _miss_out_dataset.append(_miss)
        if _miss_in_dataset:
            suggestions.append(f"You might need to set `{_miss_in_dataset}` as input. ")
        if _miss_out_dataset:
            _tmp = f"You need to provide `{_miss_out_dataset}` in DataSet and set it as input. "
            if not dataset.collater.is_empty():
                _tmp += f'Or you need to add `{_miss_out_dataset}` in the output of your collate_fn. '
            suggestions.append(_tmp)

    if check_res.unused:
        _unused = [f"\tunused field: {check_res.unused}"]
        if len(errs) > 0:
            errs.extend(_unused)
        elif check_level == STRICT_CHECK_LEVEL:
            errs.extend(_unused)

    if len(errs) > 0:
        errs.insert(0, f'Problems occurred when calling {func_signature}')
        sugg_str = ""
        if len(suggestions) > 1:
            for idx, sugg in enumerate(suggestions):
                sugg_str += f'({idx + 1}). {sugg}'
            err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        elif len(suggestions):
            sugg_str += suggestions[0]
            err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        else:
            err_str = '\n' + '\n'.join(errs)
        raise NameError(err_str)
    if _unused:
        if check_level == WARNING_CHECK_LEVEL:
            _unused_warn = _unused[0] + f' in {func_signature}.'
            warnings.warn(message=_unused_warn)


def seq_len_to_mask(seq_len, max_len=None):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::
    
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

class DummyGradScaler:
    """
    用于Dummy pytorch的GradScaler对象，防止重复写大量的if判断

    """
    def __init__(self, *args, **kwargs):
        pass

    def get_scale(self):
        return 1.0

    def is_enabled(self):
        return False

    def scale(self, outputs):
        return outputs

    def step(self, optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        pass

    def unscale_(self, optimizer):
        pass

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}

def _build_fp16_env(dummy=False):
    if dummy:
        autocast = contextlib.ExitStack
        GradScaler = DummyGradScaler
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("No cuda")
        if torch.cuda.get_device_capability(0)[0] < 7:
            warnings.warn(
                "NOTE: your device does NOT support faster training with fp16, "
                "please switch to FP32 which is likely to be faster"
            )
        try:
            from torch.cuda.amp import autocast, GradScaler
        except ImportError:
            raise RuntimeError("torch version too low (less than 1.6)")
    return autocast, GradScaler


def _is_function_contains_autocast(func):
    """
    检查func是否包含autocast，(1)是否使用了autocast的修饰器或, (2)使用使用with autocast()环境

    :param func: 待检查的函数
    """
    import re
    source = inspect.getsource(func)
    lines = source.split('\n')
    for line in lines:
        line = line.strip()
        if re.search(r'@[\w\.]*autocast\(\w*\)', line):
            raise RuntimeError("Please do not use `autocast()` decorator, use `with autocast():` instead. Please refer to"
                               " https://pytorch.org/docs/stable/notes/amp_examples.html#dataparallel-in-a-single-process ")
        if re.search(r'with [\w\.]*autocast\(\w*\):', line):
            return True
    return False


def _can_use_fp16(device, model, func):
    if parse_version(torch.__version__) < parse_version('1.6'):
        raise RuntimeError("Pytorch supports float16 after version 1.6, please upgrade your pytorch version.")
    model_device = _get_model_device(model)
    if device is None and model_device is not None and model_device.type != 'cuda':
        raise RuntimeError("You have to run in cuda device to use fp16.")
    if isinstance(device, str):
        if device=='cpu':
            raise RuntimeError("You have to run in cuda device to use fp16.")
    if isinstance(device, torch.device) and device.type=='cpu':
        raise RuntimeError("You have to run in cuda device to use fp16.")

    if (_model_contains_inner_module(model) or (isinstance(device, list) and len(device) > 1)):
        # 需要提醒用户
        if not _is_function_contains_autocast(func):
            raise RuntimeError("When use fp16 in Parallel Training, you have to set autocast() in your forward "
                               "function as described in "
                               "https://pytorch.org/docs/stable/notes/amp_examples.html#dataparallel-in-a-single-process")


def sub_column(string: str, c: int, c_size: int, title: str) -> str:
    r"""
    :param string: 要被截断的字符串
    :param c: 命令行列数
    :param c_size: instance或dataset field数
    :param title: 列名
    :return: 对一个过长的列进行截断的结果
    """
    avg = max(int(c / c_size / 2), len(title))
    string = str(string)
    res = ""
    counter = 0
    for char in string:
        if ord(char) > 255:
            counter += 2
        else:
            counter += 1
        res += char
        if counter > avg:
            res = res + "..."
            break
    return res


def pretty_table_printer(dataset_or_ins) -> PrettyTable:
    r"""
    :param dataset_or_ins: 传入一个dataSet或者instance
    ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2], field_3=["a", "b", "c"])
    +-----------+-----------+-----------------+
    |  field_1  |  field_2  |     field_3     |
    +-----------+-----------+-----------------+
    | [1, 1, 1] | [2, 2, 2] | ['a', 'b', 'c'] |
    +-----------+-----------+-----------------+
    :return: 以 pretty table的形式返回根据terminal大小进行自动截断
    """
    x = PrettyTable()
    try:
        sz = os.get_terminal_size()
        column = sz.columns
        row = sz.lines
    except OSError:
        column = 144
        row = 11

    if type(dataset_or_ins).__name__ == "DataSet":
        x.field_names = list(dataset_or_ins.field_arrays.keys())
        c_size = len(x.field_names)
        for ins in dataset_or_ins:
            x.add_row([sub_column(ins[k], column, c_size, k) for k in x.field_names])
            row -= 1
            if row < 0:
                x.add_row(["..." for _ in range(c_size)])
                break
    elif type(dataset_or_ins).__name__ == "Instance":
        x.field_names = list(dataset_or_ins.fields.keys())
        c_size = len(x.field_names)
        x.add_row([sub_column(dataset_or_ins[k], column, c_size, k) for k in x.field_names])

    else:
        raise Exception("only accept  DataSet and Instance")
    x.align = "l"

    return x

#@title fastNLP.core._parallel_utils

def parallel_apply(modules, func_name, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()
    
    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = getattr(module, func_name)(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])
    
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs

def _data_parallel_wrapper(func_name, device_ids, output_device):
    r"""
    这个函数是用于对需要多卡执行的函数的wrapper函数。参考的nn.DataParallel的forward函数

    :param str, func_name: 对network中的这个函数进行多卡运行
    :param device_ids: nn.DataParallel中的device_ids
    :param output_device: nn.DataParallel中的output_device
    :return:
    """
    
    def wrapper(network, *inputs, **kwargs):
        inputs, kwargs = scatter_kwargs(inputs, kwargs, device_ids, dim=0)
        if len(device_ids) == 1:
            return getattr(network, func_name)(*inputs[0], **kwargs[0])
        replicas = replicate(network, device_ids[:len(inputs)])
        outputs = parallel_apply(replicas, func_name, inputs, kwargs, device_ids[:len(replicas)])
        return gather(outputs, output_device)
    
    return wrapper


def _model_contains_inner_module(model):
    r"""

    :param nn.Module model: 模型文件，判断是否内部包含model.module, 多用于check模型是否是nn.DataParallel,
        nn.parallel.DistributedDataParallel。主要是在做形参匹配的时候需要使用最内部的model的function。
    :return: bool
    """
    if isinstance(model, nn.Module):
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return True
    return False