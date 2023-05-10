# @title ABCALoader
from fastNLP.io import Loader

class ABCALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        for ins in data:
            tokens = ins['review_words']
            aspects = ins['target_aspects']
            ins = Instance(raw_words=tokens, aspects=aspects)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        return ds

#@title MMABCA_withImgLoader
class MMABCALoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        for ins in data:
            tokens = ins['review_words']
            aspects = ins['target_aspects']
            img_features = torch.Tensor(ins['image_features'])

            ins = Instance(raw_words=tokens, aspects=aspects, image_features=img_features)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        return ds