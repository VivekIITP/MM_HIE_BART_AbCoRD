# MM_HIE_BART_AbCoRD
Generative (HIE-BART) Multi-modal for Aspect Span based Complaint Identification
Install the following required libraries

!pip install fitlog
!pip install fastNLP
!pip install transformers
!pip install textdistance

Before running train.py make changes to code as per the model design

from root directory import all files

import all files from "FastNLP" directory in train.py

"Modelling_bart" directory contains different implementations of BART Encoder, encoder and attention layers and model,
depending on the model we want to train we import corresponding suitable file.

In "Modelling_bart" directory:
bart_decoder.py, bart_state.py, bartseq2seq_generator.py, seq2seq_utils.py are mandatory imports for each type of model

In "Modelling_bart".bart_encoders:
There are 6 types of bart encoder implementation are present, in general best results are generated from mm_hie_encoder_mmbt.py, which is multimodal(mmbt image fusion) with hierarchical attention, other implementations vary in terms of multimodality inclusion(or not), hierarchical attention inclusion(or not) and type of fusion used mmbt/maf.

In "Modelling_bart".encoder_attn:
There are only two files, to vary model(whether to use hierarchical attention in encoder layer or not)

In "Modelling_bart".img_fusion:
There are only two files, to vary between type of image fusion to be used

In "Modelling_bart".model:
If using images then import mm_bart_model and mm_bartseq2seq_model.
