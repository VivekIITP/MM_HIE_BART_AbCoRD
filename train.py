#@title Modules
import os
import math
import random

import time
import json
import logging
import numpy as np

from typing import Union, List, Dict, Any, Optional, Tuple

import torch
import torchvision
import torch.utils.data
from torch.nn import Parameter
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather

from transformers import BartTokenizer

import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm

from transformers.activations import ACT2FN
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from transformers.utils import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ds_name = "Fashion" #@param = ["Fashion","Books", "Edible", "Electronics"]
SOURCE_MAX_LEN = 0
if ds_name in ["Fashion","Edible"]:
  SOURCE_MAX_LEN = 65

elif ds_name == "Electronics":
  SOURCE_MAX_LEN = 100
else:
  SOURCE_MAX_LEN = 80

IMG_MODEL = "vgg16" #@param = ["vgg16","vgg19","resnet152","resneXt_101_32x8d"]

from FastNLP import *
from Modelling_bart import bart_decoder, bart_state, bartseq2seq_generator, seq2seq_utils
from Modelling_bart.bart_encoders import mm_hie_encoder_mmbt
from Modelling_bart.encoder_attn import hie_bart_attn_encoder_layer
from Modelling_bart.img_fusion import img_mmbt_concat
import dataPipe, inference, loaders

#@title Get_data func
def get_data(filePath,demo):  # FOR ONLY TEXT
    pipe = DataPipe(tokenizer=bart_name, opinion_first=opinion_first,dataset=dataset)
    data_bundle = pipe.process_from_file(filePath, demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid

def get_MMdata(filePath,demo): # FOR IMG FUSION USING MMBT
    pipe = MM_DataPipe(tokenizer=bart_name, opinion_first=opinion_first,dataset=dataset)
    data_bundle = pipe.process_from_file(filePath, demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid

def get_MM_MAF_data(filePath,demo): # FOR IMG FUSION USING MAF
    pipe = MM_MAF_DataPipe(tokenizer=bart_name, opinion_first=opinion_first,dataset=dataset)
    data_bundle = pipe.process_from_file(filePath, demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid

#@title Constants
lr = 5e-5
num_beams = 4
length_penalty = 1.0
decoder_type = 'avg_score'
bart_name = 'facebook/bart-base'
use_encoder_mlp = True
use_last_layer_attention = True

demo = False
model_path = "/Model_output/" # output path

fashion_category = ['<<color>>', '<<fit>>', '<<misc>>', '<<price>>', '<<quality>>', '<<service>>', '<<style>>']
fashion_category_key = ['color', 'fit', 'misc', 'price', 'quality', 'service', 'style']
fashion_category_dict = dict(zip(fashion_category_key, fashion_category))

books_category = ['<<content>>','<<price>>', '<<quality>>', '<<service>>']
books_category_key = ['content', 'price', 'quality', 'service']
books_category_dict = dict(zip(books_category_key, books_category))

edible_category = ['<<quality>>', '<<service>>', '<<taste>>', '<<smell>>', '<<price>>']
edible_category_key = ['quality', 'service', 'taste', 'smell', 'price']
edible_category_dict = dict(zip(edible_category_key, edible_category))

electronics_category = ['<<design>>', '<<software>>', '<<hardware>>', '<<price>>', '<<quality>>', '<<service>>']
electronics_category_key = ['design', 'software', 'hardware', 'price', 'quality', 'service']
electronics_category_dict = dict(zip(electronics_category_key, electronics_category))

category_dict = None
if ds_name == "Fashion":
  category_dict = fashion_category_dict
elif ds_name == "Books":
  category_dict = books_category_dict
elif ds_name == "Edible":
  category_dict = edible_category_dict
else:
  category_dict = electronics_category_dict

def cmp_aspect(v1, v2):
    return v1['from'] - v2['from']

#@title Preparing Data
# data_bundle, tokenizer, mapping2id,mapping2targetid = get_data(f'/content/drive/MyDrive/Datasets/'+ds_name+'/json_text/',False) # FOR ONLY TEXT
data_bundle, tokenizer, mapping2id,mapping2targetid = get_MMdata(f'/content/drive/MyDrive/Datasets/'+ds_name,False) # FOR IMG FUSION USING MMBT
# data_bundle, tokenizer, mapping2id,mapping2targetid = get_MM_MAF_data(f'/content/drive/MyDrive/Datasets/'+ds_name,False) # FOR IMG FUSION USING MAF

# FOR FEW SHOT
# data_bundle, tokenizer, mapping2id,mapping2targetid = get_MMdata(f'/content/drive/MyDrive/BART/Datasets/Electronics/',False)
# raw_words_test = data_bundle.datasets['dev']['raw_words'].__dict__['content']

max_len = 20
max_len_a = 0.9 

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

idtarget2map=inv_map = {v: k for k, v in mapping2targetid.items()}

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())
vocab_size = len(tokenizer)

print(idtarget2map)
print(label_ids)
print(vocab_size)

raw_words_test = data_bundle.datasets['test']['raw_words'].__dict__['content']

data_bundle.delete_field(field_name='aspects')
data_bundle.delete_field(field_name='raw_words')
# data_bundle.datasets['train']

#@title modeling
n_epochs = 50
batch_size = 4

## IF USING ONLY TEXT UNCOMMENT THE BELOW CODE AND COMMENT THE BELOW MULTIMODAL CODE
# model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=None,
#                                         copy_gate=False, use_encoder_mlp=use_encoder_mlp,use_last_layer_attention=use_last_layer_attention, use_recur_pos=False)

# print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))

# model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
#                                 eos_token_id=eos_token_id,
#                                 max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
#                                 repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
#                                 restricter=None)

## MULTIMODAL CODE
model = MMBartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=None,
                                        copy_gate=False, use_encoder_mlp=use_encoder_mlp,use_last_layer_attention=use_last_layer_attention, use_recur_pos=False)

print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))

model = MMSequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                                repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                                restricter=None)

parameters = []
params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
parameters.append(params)

params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr':lr, 'weight_decay':0}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

#@title optimizer,sampler,metric,callbacks
optimizer = optim.AdamW(parameters)
    
callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
fitlog.set_log_dir('caches')

sampler = BucketSampler(num_buckets=4, batch_size = batch_size,seq_len_field_name='src_seq_len')
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids))

#@title training with val eval
trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                      loss=Seq2SeqLoss(),
                      batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                      num_workers=2, n_epochs=n_epochs, print_every=1,
                      dev_data=data_bundle.get_dataset('dev'), metrics=metric, metric_key='quad_f',
                      validate_every=-1, save_path=model_path, use_tqdm=True, device=device,
                      callbacks=callbacks, check_code_level=0, test_use_tqdm=False,
                      test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size, verbose=1)

trainer.train(load_best_model=True)

#@title Saving trained model
torch.save(model, 'modelpath.pth')