#@title DataPipe
from itertools import chain
from functools import cmp_to_key
from transformers import BartTokenizer
from fastNLP.io import Pipe, DataBundle, Loader

#@title Adding DataBundle additional methods
def set_input(DataBundle, *field_names, flag=True, use_1st_ins_infer_dim_type=True, ignore_miss_dataset=True):
        for field_name in field_names:
            for name, dataset in DataBundle.datasets.items():
                if not ignore_miss_dataset and not dataset.has_field(field_name):
                    raise KeyError(f"Field:{field_name} was not found in DataSet:{name}")
                if not dataset.has_field(field_name):
                    continue
                else:
                    dataset.set_input(field_name, flag=flag, use_1st_ins_infer_dim_type=use_1st_ins_infer_dim_type)
        return DataBundle

setattr(DataBundle, "set_input", set_input)

def set_target(DataBundle, *field_names, flag=True, use_1st_ins_infer_dim_type=True, ignore_miss_dataset=True):
        for field_name in field_names:
            for name, dataset in DataBundle.datasets.items():
                if not ignore_miss_dataset and not dataset.has_field(field_name):
                    raise KeyError(f"Field:{field_name} was not found in DataSet:{name}")
                if not dataset.has_field(field_name):
                    continue
                else:
                    dataset.set_target(field_name, flag=flag, use_1st_ins_infer_dim_type=use_1st_ins_infer_dim_type)
        return DataBundle

setattr(DataBundle, "set_target", set_target)

def set_pad_val(DataBundle, field_name, pad_val, ignore_miss_dataset=True):
        for name, dataset in DataBundle.datasets.items():
            if dataset.has_field(field_name=field_name):
                dataset.set_pad_val(field_name=field_name, pad_val=pad_val)
            elif not ignore_miss_dataset:
                raise KeyError(f"{field_name} not found DataSet:{name}.")
        return DataBundle

setattr(DataBundle, "set_pad_val", set_pad_val)


def set_ignore_type(DataBundle, *field_names, flag=True, ignore_miss_dataset=True):
        for name, dataset in DataBundle.datasets.items():
            for field_name in field_names:
                if dataset.has_field(field_name=field_name):
                    dataset.set_ignore_type(field_name, flag=flag)
                elif not ignore_miss_dataset:
                    raise KeyError(f"{field_name} not found DataSet:{name}.")
        return DataBundle

setattr(DataBundle, "set_ignore_type", set_ignore_type)


def cmp_aspect(v1, v2):
    return v1['from'] - v2['from']


class DataPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base'):
        super(DataPipe, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_tokens("<<no_cause>>")
        self.mapping = category_dict
        self.mapping.update({'COMP': '<<complaint>>', 'NON-COMP': '<<non_complaint>>'})

        cur_num_tokens = self.tokenizer.vocab_size
        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)

        for tok in sorted_add_tokens:
                    assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id

        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
                
        self.mapping2id = {}
        self.mapping2targetid = {}

        self.src_max_len = SOURCE_MAX_LEN - 1

        for key, value in self.mapping.items():
                    key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
                    assert len(key_id) == 1, value
                    assert key_id[0] >= cur_num_tokens
                    self.mapping2id[key] = key_id[0]
                    self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'nature': str
            'category':str
        }],
        输出为[o_s, o_e, a_s, a_e, c]或者[a_s, a_e, o_s, o_e, c]
        :param data_bundle:
        :return:
        """
        target_shift = len(self.mapping) + 2  # It is because the first digit is sos, followed by eos, and then

        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)

            src_input_ids = list(chain(*word_bpes))
            curr_len = len(src_input_ids)
            
            if curr_len < self.src_max_len:
              pad_no = self.src_max_len - curr_len
              for i in range(pad_no):
                  src_input_ids.append(self.tokenizer.pad_token_id)
            else: # >=
              src_input_ids = src_input_ids[:self.src_max_len]

            src_input_ids.append(self.tokenizer.eos_token_id)
            word_bpes.append([self.tokenizer.eos_token_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # special start
            target_spans = []
            _word_bpes = list(chain(*word_bpes))
            aspects = ins['aspects']
            aspects = sorted(aspects, key=cmp_to_key(cmp_aspect))
            for aspect_ins in aspects:
                start_index = aspect_ins['from']
                end_index = aspect_ins['to']-1
                if cum_lens[end_index] > self.src_max_len:
                  continue
                a_start_bpe = cum_lens[start_index]  # because there is a sos shift
                a_end_bpe = cum_lens[end_index]  # Here, since it was an open interval before, it happened to be the lastone word of the beginning
                target_spans.append([a_start_bpe+target_shift, a_end_bpe+target_shift, self.mapping2targetid[aspect_ins['category']]+2])
                target_spans[-1].append(self.mapping2targetid[aspect_ins['nature']]+2) 
                target_spans[-1] = tuple(target_spans[-1])

            target.extend(list(chain(*target_spans)))
            target.append(1)  # append 1 is due to special eos
            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': src_input_ids}

        data_bundle.apply_more(prepare_target)#, tqdm_desc='Pre. tgt.', use_tqdm=True)

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = ABCALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle

#@title MM_DataPipe ImgConcat & fixed length
class MM_DataPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base'):
        super(MM_DataPipe, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)
        self.additional_special_tokens = [ "<<no_cause>>", "<<img_feat>>"]

        self.mapping = category_dict
        self.mapping.update({'COMP': '<<complaint>>', 'NON-COMP': '<<non_complaint>>'})
        
        cur_num_tokens = self.tokenizer.vocab_size
        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)

        for tok in sorted_add_tokens:
                    assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.additional_special_tokens.extend(sorted_add_tokens)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': self.additional_special_tokens}
        )
                
        self.mapping2id = {}
        self.mapping2targetid = {}

        self.src_max_len = SOURCE_MAX_LEN - 3 # for sep_id, img_id, eos_id

        for key, value in self.mapping.items():
                    key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
                    assert len(key_id) == 1, value
                    assert key_id[0] >= cur_num_tokens
                    self.mapping2id[key] = key_id[0]
                    self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        
        """
        target_shift = len(self.mapping) + 2  # It is because the first digit is sos, followed by eos, and then

        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            
            src_input_ids = list(chain(*word_bpes))
            curr_len = len(src_input_ids)
            
            if curr_len < self.src_max_len:
              pad_no = self.src_max_len - curr_len
              for i in range(pad_no):
                  src_input_ids.append(self.tokenizer.pad_token_id)
            else: # >=
              src_input_ids = src_input_ids[:self.src_max_len]

            src_input_ids.append(self.tokenizer.sep_token_id)
            src_input_ids.append(self.tokenizer.convert_tokens_to_ids("<<img_feat>>"))
            src_input_ids.append(self.tokenizer.eos_token_id)
            word_bpes.append([self.tokenizer.sep_token_id])
            word_bpes.append([self.tokenizer.convert_tokens_to_ids("<<img_feat>>")])
            word_bpes.append([self.tokenizer.eos_token_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # special start
            target_spans = []
            _word_bpes = list(chain(*word_bpes))
            aspects = ins['aspects']
            aspects = sorted(aspects, key=cmp_to_key(cmp_aspect))
            for aspect_ins in aspects:
                start_index = aspect_ins['from']
                end_index = aspect_ins['to']-1
                if cum_lens[end_index] > self.src_max_len:
                  continue
                a_start_bpe = cum_lens[start_index]  # because there is a sos shift
                a_end_bpe = cum_lens[end_index]  # Here, since it was an open interval before, it happened to be the lastone word of the beginning
                target_spans.append([a_start_bpe+target_shift, a_end_bpe+target_shift, self.mapping2targetid[aspect_ins['category']]+2])
                target_spans[-1].append(self.mapping2targetid[aspect_ins['nature']]+2) 
                target_spans[-1] = tuple(target_spans[-1])

            target.extend(list(chain(*target_spans)))
            target.append(1)  # append 1 is due to special eos
            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': src_input_ids, 'image_features': ins['image_features']}

        data_bundle.apply_more(prepare_target)#, tqdm_desc='Pre. tgt.', use_tqdm=True)

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'image_features', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = MMABCALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle

#@title MM_MAF_DataPipe ImgContextAttn & fixed length src
class MM_MAF_DataPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base'):
        super(MM_MAF_DataPipe, self).__init__()
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)
        self.additional_special_tokens = [ "<<no_cause>>"]

        self.mapping = category_dict
        self.mapping.update({'COMP': '<<complaint>>', 'NON-COMP': '<<non_complaint>>'})
        
        cur_num_tokens = self.tokenizer.vocab_size
        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)

        for tok in sorted_add_tokens:
                    assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.additional_special_tokens.extend(sorted_add_tokens)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': self.additional_special_tokens}
        )
        self.src_max_len = SOURCE_MAX_LEN - 1 # for sos
                
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
                    key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
                    assert len(key_id) == 1, value
                    assert key_id[0] >= cur_num_tokens
                    self.mapping2id[key] = key_id[0]
                    self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        
        """
        target_shift = len(self.mapping) + 2  # It is because the first digit is sos, followed by eos, and then

        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            for word in raw_words:
                  bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                  bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                  word_bpes.append(bpes)

            src_input_ids = list(chain(*word_bpes))
            curr_len = len(src_input_ids)
            
            if curr_len < self.src_max_len:
              pad_no = self.src_max_len - curr_len
              for i in range(pad_no):
                  src_input_ids.append(self.tokenizer.pad_token_id)
            else: # >=
              src_input_ids = src_input_ids[:self.src_max_len]

            src_input_ids.append(self.tokenizer.eos_token_id)
            word_bpes.append([self.tokenizer.eos_token_id])
                        
            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # special start
            target_spans = []
            # _word_bpes = list(chain(*word_bpes))
            aspects = ins['aspects']
            aspects = sorted(aspects, key=cmp_to_key(cmp_aspect))
            for aspect_ins in aspects:
                start_index = aspect_ins['from']
                end_index = aspect_ins['to']-1
                if cum_lens[end_index] > self.src_max_len:
                  continue 
                a_start_bpe = cum_lens[start_index]  # because there is a sos shift
                a_end_bpe = cum_lens[start_index]  # Here, since it was an open interval before, it happened to be the lastone word of the beginning
                target_spans.append([a_start_bpe+target_shift, a_end_bpe+target_shift, self.mapping2targetid[aspect_ins['category']]+2])
                target_spans[-1].append(self.mapping2targetid[aspect_ins['nature']]+2) 
                target_spans[-1] = tuple(target_spans[-1])

            target.extend(list(chain(*target_spans)))
            target.append(1)  # append 1 is due to special eos
            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': src_input_ids, 'image_features': ins['image_features']}

        data_bundle.apply_more(prepare_target)#, tqdm_desc='Pre. tgt.', use_tqdm=True)

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'image_features', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """
        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = MMABCALoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)
        # data_bundle = self.process(temp_data_bundle)

        return data_bundle