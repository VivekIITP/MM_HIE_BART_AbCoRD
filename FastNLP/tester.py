#@title fastNLP.core.Tester
from functools import cmp_to_key, partial
from tqdm.auto import tqdm

class Tester(object):
    r"""
    Tester
    It is a class for performance testing when data, models and metrics are provided. 
    The model, data and metric need to be passed in for verification.
    """
    
    def __init__(self, data, model, metrics, batch_size=16, num_workers=0, device=None, verbose=1, use_tqdm=True,
                 fp16=False, **kwargs):
        r"""
        
        :param ~fastNLP.DataSet,~fastNLP.BatchIter data: Datasets to test
        :param torch.nn.Module model: model used
        :param ~fastNLP.core.metrics.MetricBase,List[~fastNLP.core.metrics.MetricBase] metrics: metrics used during testing
        :param int batch_size: How big is the batch_size used in evaluation.
        :param str,int,torch.device,list(int) device: which device to load the model to. The default is None, that is, the Trainer is not correct for the model
            Computing location for management. The following inputs are supported:
    
            1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] in order of 'cpu', the first visible GPU, the first visible GPU In, the visible second GPU;
    
            2. torch.device: Load the model onto torch.device.
    
            3. int: the gpu whose device_id is this value will be used for training
    
            4. list(int): If there is more than one device, torch.nn.DataParallel will be used to wrap the model, and the incoming device will be used.
    
            5. None. If it is None, no processing will be performed on the model. If the incoming model is torch.nn.DataParallel, the value must be None.
    
            If the model is predicted by predict(), it will not be able to use the multi-card (DataParallel) for verification, and only the model on the first card will be used.
        :param int verbose: If it is 0, no information will be output; if it is 1, the verification result will be printed.
        :param bool use_tqdm: Whether to use tqdm to display test progress; if False, nothing will be displayed.
        :param bool fp16: Whether to use float16 for verification
        :param kwargs:
            Sampler sampler: Support passing in sampler to control the test sequence
            bool pin_memory: Whether to use pin memory for the generated tensor, which may speed up the data speed。
        """
        super(Tester, self).__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")
        
        self.metrics = _prepare_metrics(metrics)
        
        self.data = data
        self._model = _move_model_to_device(model, device=device)
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.logger = logger
        self.pin_memory = kwargs.get('pin_memory', True)

        if isinstance(data, DataSet):
            sampler = kwargs.get('sampler', None)
            if sampler is None:
                sampler = SequentialSampler()
            elif not isinstance(sampler, (Sampler, torch.utils.data.Sampler)):
                raise ValueError(f"The type of sampler should be fastNLP.BaseSampler or pytorch's Sampler, got {type(sampler)}")
            if hasattr(sampler, 'set_batch_size'):
                sampler.set_batch_size(batch_size)
            self.data_iterator = DataSetIter(dataset=data, batch_size=batch_size, sampler=sampler,
                                             num_workers=num_workers,
                                             pin_memory=self.pin_memory)
        elif isinstance(data, BatchIter):
            self.data_iterator = data
        else:
            raise TypeError("data type {} not support".format(type(data)))

        # check predict
        if (hasattr(self._model, 'predict') and callable(self._model.predict)) or \
                (_model_contains_inner_module(self._model) and hasattr(self._model.module, 'predict') and
                 callable(self._model.module.predict)):
            if isinstance(self._model, nn.DataParallel):
                self._predict_func_wrapper = partial(_data_parallel_wrapper('predict',
                                                                    self._model.device_ids,
                                                                    self._model.output_device),
                                                     network=self._model.module)
                self._predict_func = self._model.module.predict  # for matching parameters
            elif isinstance(self._model, nn.parallel.DistributedDataParallel):
                self._predict_func = self._model.module.predict
                self._predict_func_wrapper = self._model.module.predict  # for calling
            else:
                self._predict_func = self._model.predict
                self._predict_func_wrapper = self._model.predict
        else:
            if _model_contains_inner_module(self._model):
                self._predict_func_wrapper = self._model.forward
                self._predict_func = self._model.module.forward
            else:
                self._predict_func = self._model.forward
                self._predict_func_wrapper = self._model.forward

        if fp16:
            _can_use_fp16(model=model, device=device, func=self._predict_func)
        self.auto_cast, _grad_scaler = _build_fp16_env(not fp16)


    def test(self):
        r"""Start the verification and return the verification result.

        :return Dict[Dict]: Two-layer nested structure of dict, the first layer of dict is the name of the metric; 
        the second layer is the index of this metric. An example of AccuracyMetric is {'AccuracyMetric': {'acc': 1.0}}.
        """
        # turn on the testing mode; clean up the history
        self._model_device = _get_model_device(self._model)
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = self.data_iterator
        eval_results = {}
        try:
            with torch.no_grad():
                if not self.use_tqdm:
                    inner_tqdm = _pseudo_tqdm
                else:
                    inner_tqdm = tqdm
                with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                    pbar.set_description_str(desc="Test")

                    start_time = time.time()

                    for batch_x, batch_y in data_iterator:
                        _move_dict_value_to_device(batch_x, batch_y, device=self._model_device,
                                                   non_blocking=self.pin_memory)
                        with self.auto_cast():
                            pred_dict = self._data_forward(self._predict_func, batch_x)
                            if not isinstance(pred_dict, dict):
                                raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                                f"must be `dict`, got {type(pred_dict)}.")
                            for metric in self.metrics:
                                metric(pred_dict, batch_y)

                        if self.use_tqdm:
                            pbar.update()

                    for metric in self.metrics:
                        eval_result = metric.get_metric()
                        if not isinstance(eval_result, dict):
                            raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                            f"`dict`, got {type(eval_result)}")
                        metric_name = metric.get_metric_name()
                        eval_results[metric_name] = eval_result
                    pbar.close()
                    end_time = time.time()
                    test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                    if self.verbose >= 0:
                        self.logger.info(test_str)
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)
        finally:
            self._mode(network, is_test=False)
        if self.verbose >= 1:
            logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
        return eval_results

    
    def _mode(self, model, is_test=False):
        r"""Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()
    
    def _data_forward(self, func, x):
        r"""A forward pass of the model. """
        x = _build_args(func, **x)
        y = self._predict_func_wrapper(**x)
        return y
    
    def _format_eval_results(self, results):
        r"""Override this method to support more print formats.

        :param results: dict, (str: float) is (metrics name: value)

        """
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + ': '
            _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
            _str += '\n'
        return _str[:-1]