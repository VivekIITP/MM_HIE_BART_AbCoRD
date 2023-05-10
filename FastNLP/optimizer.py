#@title from fastNLP.core import Optimizer
class Optimizer(object):
    r"""
    Optimizer
    """
    
    def __init__(self, model_params, **kwargs):
        r"""
        
        :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
        :param kwargs: additional parameters.
        """
        if model_params is not None and not hasattr(model_params, "__next__"):
            raise RuntimeError("model parameters should be a generator, rather than {}.".format(type(model_params)))
        self.model_params = model_params
        self.settings = kwargs

    
    def construct_from_pytorch(self, model_params):
        raise NotImplementedError

    @staticmethod
    def _get_require_grads_param(params):
        r"""
        将params中不需要gradient的删除
        
        :param iterable params: parameters
        :return: list(nn.Parameters)
        """
        return [param for param in params if param.requires_grad]