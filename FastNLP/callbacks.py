#@title from fastNLP.core.callback import *(GradientClipCallback, WarmupCallback)
import fitlog
import deepcopy

class Callback(object):
    r"""
    Callback是fastNLP中被设计用于增强 :class:`~fastNLP.Trainer` 的类。
    如果Callback被传递给了 Trainer , 则 Trainer 会在对应的阶段调用Callback的函数，
    具体调用时机可以通过 :mod:`trainer 模块<fastNLP.core.trainer>` 查看。
    这是Callback的基类，所有的callback必须继承自这个类

    """
    
    def __init__(self):
        super(Callback, self).__init__()
        self._trainer = None  # 在Trainer内部被重新赋值
        self._disabled = False

    def __repr__(self):
        return self.__class__.__name__

    @property
    def trainer(self):
        r"""
        该属性可以通过self.trainer获取到，一般情况下不需要使用这个属性。
        """
        return self._trainer

    @property
    def grad_scaler(self):
        r"""
        float16的gradient scaler
        """
        return self._trainer.grad_scaler

    @property
    def auto_cast(self):
        r"""
        float16用的auto cast环境
        """
        return self._trainer.auto_cast
    
    @property
    def step(self):
        r"""当前运行到的step, 范围为[1, self.n_steps+1)"""
        return self._trainer.step
    
    @property
    def n_steps(self):
        r"""Trainer一共会采多少个batch。当Trainer中update_every设置为非1的值时，该值不等于update的次数"""
        return self._trainer.n_steps
    
    @property
    def batch_size(self):
        r"""train和evaluate时的batch_size为多大"""
        return self._trainer.batch_size
    
    @property
    def epoch(self):
        r"""当前运行的epoch数，范围是[1, self.n_epochs+1)"""
        return self._trainer.epoch
    
    @property
    def n_epochs(self):
        r"""一共会运行多少个epoch"""
        return self._trainer.n_epochs
    
    @property
    def optimizer(self):
        r"""初始化Trainer时传递的Optimizer"""
        return self._trainer.optimizer
    
    @property
    def model(self):
        r"""正在被Trainer训练的模型"""
        return self._trainer.model
    
    @property
    def pbar(self):
        r"""如果在Callback中需要打印内容，请使用self.pbar.write(str)。否则可能出现命令行显示效果不太好的问题。在
        on_train_begin(), on_train_end(), on_exception()中请不要使用该属性，通过print输出即可。"""
        return self._trainer.pbar
    
    @property
    def update_every(self):
        r"""Trainer中的模型多少次反向传播才进行一次梯度更新，在Trainer初始化时传入的。"""
        return self._trainer.update_every
    
    @property
    def batch_per_epoch(self):
        r"""每个epoch一共有多少个batch，只有在on_epoch_begin之后才能调用该属性。"""
        return self._trainer.batch_per_epoch

    @property
    def is_master(self):
        return self._trainer.is_master

    @property
    def disabled(self):
        return self._disabled

    @property
    def logger(self):
        return getattr(self._trainer, 'logger', logger)

    def on_train_begin(self):
        r"""
        在Train过程开始之前调用。

        :return:
        """
        pass

    
    def on_epoch_begin(self):
        r"""
        在每个epoch开始之前调用一次

        :return:
        """
        pass

    
    def on_batch_begin(self, batch_x, batch_y, indices):
        r"""
        每次采集到一个batch的数据则调用一次。这里对batch_x或batch_y删除添加内容是可以影响到Trainer中内容的。所以在这一步
        可以进行一些负采样之类的操作。batch_x和batch_y中的tensor已经被放置到了模型所在的设备上。

        :param dict batch_x: DataSet中被设置为input的field的batch。
        :param dict batch_y: DataSet中被设置为target的field的batch。
        :param list(int) indices: 这次采样使用到的indices，可以通过DataSet[indices]获取出这个batch采出的Instance，在一些
            情况下可以帮助定位是哪个Sample导致了错误。仅当num_workers=0时有效。
        :return:
        """
        pass

    
    def on_loss_begin(self, batch_y, predict_y):
        r"""
        在计算loss前调用，即这里修改batch_y或predict_y的值是可以影响到loss计算的。

        :param dict batch_y: 在DataSet中被设置为target的field的batch集合。
        :param dict predict_y: 模型的forward()返回的结果。
        :return:
        """
        pass

    
    def on_backward_begin(self, loss):
        r"""
        在loss得到之后，但在反向传播之前。可能可以进行loss是否为NaN的检查。

        :param torch.Tensor loss: 计算得到的loss值
        :return:
        """
        pass

    
    def on_backward_end(self):
        r"""
        反向梯度传播已完成，但由于update_every的设置，可能并不是每一次调用都有梯度。到这一步，还没有更新参数。

        :return:
        """
        pass

    
    def on_step_end(self):
        r"""
        到这里模型的参数已经按照梯度更新。但可能受update_every影响，并不是每次都更新了。

        :return:
        """
        pass

    
    def on_batch_end(self):
        r"""
        这一步与on_step_end是紧接着的。只是为了对称性加上了这一步。

        """
        pass

    
    def on_valid_begin(self):
        r"""
        如果Trainer中设置了验证，则发生验证前会调用该函数

        :return:
        """
        pass

    
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        r"""
        每次执行验证集的evaluation后会调用。

        :param Dict[str: Dict[str: float]] eval_result: , evaluation的结果。一个例子为{'AccuracyMetric':{'acc':1.0}}，即
            传入的dict是有两层，第一层是metric的名称，第二层是metric的具体指标。
        :param str metric_key: 初始化Trainer时传入的metric_key。
        :param torch.Optimizer optimizer: Trainer中使用的优化器。
        :param bool is_better_eval: 当前dev结果是否比之前的好。
        :return:
        """
        pass

    
    def on_epoch_end(self):
        r"""
        每个epoch结束将会调用该方法
        """
        pass

    
    def on_train_end(self):
        r"""
        训练结束，调用该方法
        """
        pass

    
    def on_exception(self, exception):
        r"""
        当训练过程出现异常，会触发该方法
        :param exception: 某种类型的Exception，比如KeyboardInterrupt等
        """
        pass


def _transfer(func):
    r"""装饰器，将对CallbackManager的调用转发到各个Callback子类.
    
    :param func:
    :return:
    """
    
    def wrapper(manager, *arg):
        returns = []
        for callback in manager.callbacks:
            if callback.disabled:
                continue
            returns.append(getattr(callback, func.__name__)(*arg))
        return returns
    
    return wrapper

class CallbackManager(Callback):
    r"""
    内部使用的Callback管理类
    """
    def __init__(self, env, callbacks=None):
        r"""

        :param dict env: The key is the name of the Trainer attribute(str). The value is the attribute itself.
        :param List[Callback] callbacks:
        """
        super(CallbackManager, self).__init__()
        # set attribute of trainer environment
        self._env = env
        self.callbacks = []
        if callbacks:
            self.callbacks = self.prepare_callbacks(callbacks)

    def prepare_callbacks(self, callbacks):
        if not callbacks:
            return []
        if isinstance(callbacks, list):
            if all([isinstance(cb, Callback) for cb in callbacks]) is True:
                pass
            else:
                obj = [not isinstance(cb, Callback) for cb in callbacks][0]
                raise TypeError(f"Expect sub-classes of Callback. Got {type(obj)}")
        else:
            raise TypeError(f"Expect callbacks in CallbackManager(callbacks) to be list. Got {type(callbacks)}.")

        for env_name, env_val in self._env.items():
            for callback in callbacks:
                setattr(callback, '_' + env_name, env_val)  # Callback.trainer
        return callbacks

    @_transfer
    def on_train_begin(self):
        pass
    
    @_transfer
    def on_epoch_begin(self):
        pass
    
    @_transfer
    def on_batch_begin(self, batch_x, batch_y, indices):
        pass
    
    @_transfer
    def on_loss_begin(self, batch_y, predict_y):
        pass
    
    @_transfer
    def on_backward_begin(self, loss):
        pass
    
    @_transfer
    def on_backward_end(self):
        pass
    
    @_transfer
    def on_step_end(self):
        pass
    
    @_transfer
    def on_batch_end(self):
        pass
    
    @_transfer
    def on_valid_begin(self):
        pass
    
    @_transfer
    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        pass

    @_transfer
    def on_validation(self):
        pass
    
    @_transfer
    def on_epoch_end(self):
        pass
    
    @_transfer
    def on_train_end(self):
        pass
    
    @_transfer
    def on_exception(self, exception):
        pass


class GradientClipCallback(Callback):
    r"""
    每次backward前，将parameter的gradient clip到某个范围。
    """
    
    def __init__(self, parameters=None, clip_value=1, clip_type='norm'):
        r"""
        
        :param None,torch.Tensor,List[torch.Tensor] parameters: 一般通过model.parameters()获得。
            如果为None则默认对Trainer的model中所有参数进行clip
        :param float clip_value: 将gradient 限制到[-clip_value, clip_value]。clip_value应该为正数
        :param str clip_type: 支持'norm', 'value'
            两种::
    
                1 'norm', 将gradient的norm rescale到[-clip_value, clip_value]
            
                2 'value', 将gradient限制在[-clip_value, clip_value],
                    小于-clip_value的gradient被赋值为-clip_value;
                    大于clip_value的gradient被赋值为clip_value.
        """
        super().__init__()
        
        from torch import nn
        if clip_type == 'norm':
            self.clip_fun = nn.utils.clip_grad_norm_
        elif clip_type == 'value':
            self.clip_fun = nn.utils.clip_grad_value_
        else:
            raise ValueError("Only supports `norm` or `value` right now.")
        if parameters is not None:
            self.parameters = list(parameters)
        else:
            self.parameters = None
        self.clip_value = clip_value

    
    def on_backward_end(self):
        if self.step%self.update_every==0:
            if self.trainer.fp16:
                self.grad_scaler.unscale_(self.optimizer)
            if self.parameters is not None:
                self.clip_fun(self.parameters, self.clip_value)
            else:
                self.clip_fun(self.model.parameters(), self.clip_value)


class WarmupCallback(Callback):
    r"""
    learning rate按照一定的速率从0上升到设置的learning rate。
    """
    def __init__(self, warmup=0.1, schedule='constant'):
        r"""
        
        :param int,float warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
            如0.1, 则前10%的step是按照schedule策略调整learning rate。
        :param str schedule: 以哪种方式调整。
            linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
            constant前warmup的step上升到指定learning rate，后面的step保持learning rate.
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")


    def _get_constant_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress<self.warmup:
            return progress/self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def on_train_begin(self):
        self.t_steps = (len(self.trainer.train_data) // (self.batch_size*self.update_every) +
                            int(len(self.trainer.train_data) % (self.batch_size*self.update_every)!= 0)) * self.n_epochs
        if self.warmup>1:
            self.warmup = self.warmup/self.t_steps
        self.t_steps = max(2, self.t_steps)  # 不能小于2
        # 获取param_group的初始learning rate
        for group in self.optimizer.param_groups:
            self.initial_lrs.append(group['lr'])

    def on_backward_end(self):
        if self.step%self.update_every==0:
            progress = (self.step/self.update_every)/self.t_steps
            for lr, group in zip(self.initial_lrs, self.optimizer.param_groups):
                group['lr'] = lr * self.get_lr(progress)


class FitlogCallback(Callback):
    r"""
    该callback可将loss和progress写入到fitlog中; 如果Trainer有dev的数据，将自动把dev的结果写入到log中; 同时还支持传入
    一个(或多个)test数据集进行测试(只有在trainer具有dev时才能使用)，每次在dev上evaluate之后会在这些数据集上验证一下。
    并将验证结果写入到fitlog中。这些数据集的结果是根据dev上最好的结果报道的，即如果dev在第3个epoch取得了最佳，则
    fitlog中记录的关于这些数据集的结果就是来自第三个epoch的结果。
    """

    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=1, log_exception=False):
        r"""
        
        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用多个Trainer中的metric对数据进行验证。如果需要
            传入多个DataSet请通过dict的方式传入，dict的key将作为对应dataset的name传递给fitlog。data的结果的名称以'data'开头。
        :param ~fastNLP.Tester,Dict[~fastNLP.Tester] tester: Tester对象，将在on_valid_end时调用。tester的结果的名称以'tester'开头
        :param int log_loss_every: 多少个step记录一次loss(记录的是这几个batch的loss平均值)，如果数据集较大建议将该值设置得
            大一些，不然会导致log文件巨大。默认为0, 即不要记录loss。
        :param int verbose: 是否在终端打印evaluation的结果，0不打印。
        :param bool log_exception: fitlog是否记录发生的exception信息
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self._log_exception = log_exception
        assert isinstance(log_loss_every, int) and log_loss_every>=0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")
        
        self.verbose = verbose
        self._log_loss_every = log_loss_every
        self._avg_loss = 0

    def on_train_begin(self):
        print(len(self.datasets), len(self.testers))
        if (len(self.datasets) > 0 or len(self.testers) > 0) and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra data to do evaluation.")
        
        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.trainer.batch_size),
                                metrics=self.trainer.metrics,
                                verbose=0,
                                use_tqdm=self.trainer.kwargs.get('test_use_tqdm', self.trainer.use_tqdm),
                                sampler=self.trainer.kwargs.get('test_sampler', None))
                self.testers[key] = tester
        fitlog.add_progress(total_steps=self.n_steps)
    
    def on_backward_begin(self, loss):
        if self._log_loss_every>0:
            self._avg_loss += loss.item()
            if self.step%self._log_loss_every==0:
                fitlog.add_loss(self._avg_loss/self._log_loss_every*self.update_every, name='loss', step=self.step, epoch=self.epoch)
                self._avg_loss = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose != 0:
                        self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception as e:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        fitlog.finish()
    
    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


class CallbackException(BaseException):
    r"""
   当需要通过callback跳出训练的时候可以通过抛出CallbackException并在on_exception中捕获这个值。
   """
    
    def __init__(self, msg):
        r"""
        
        :param str msg: Exception的信息。
        """
        super(CallbackException, self).__init__(msg)