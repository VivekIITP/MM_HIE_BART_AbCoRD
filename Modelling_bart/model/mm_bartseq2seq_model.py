#@title MMBartSeq2SeqModel

class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, MultiModalBartEncoder)
        self.bart_encoder = encoder

    def forward(self, image_features, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, image_features=image_features, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states

        
class MMBartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None, copy_gate=False,
                    use_encoder_mlp=False,use_last_layer_attention=False, use_recur_pos=False, tag_first=False):
        # model = BartModel.from_pretrained(bart_model)
        model = MultiModalBartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder

        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        label_ids = sorted(label_ids)
        if decoder_type is None:
            assert copy_gate is False
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type =='avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                              use_encoder_mlp=use_encoder_mlp,use_last_layer_attention=use_last_layer_attention)
        else:
            raise RuntimeError("Unsupported feature.")

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, image_features, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(image_features, src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(self, image_features, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):
        """
        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(image_features, src_tokens, src_seq_len, first, tgt_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")

class MMSequenceGeneratorModel(nn.Module):
    """
    Used to encapsulate Seq2SeqModel so that it can do generation tasks
    """

    def __init__(self, seq2seq_model: Seq2SeqModel, bos_token_id, eos_token_id=None, max_length=30, max_len_a=0.0,
                 num_beams=1, do_sample=True,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0,
                 restricter=None):
        """
        :param Seq2SeqModel seq2seq_model: sequence-to-sequence model. It will be generated using the decoder of seq2seq_model
        :param int,None bos_token_id: token id at the beginning of the sentence
        :param int,None eos_token_id: token id at the end of the sentence
        :param int max_length: The maximum length of the generated sentence, the decode length of each sentence is max_length + max_len_a*src_len
        :param float max_len_a: The decoded length of each sentence is max_length + max_len_a*src_len. If it is not 0, you need to ensure that encoder_mask is included in State
        :param int num_beams: beam search size
        :param bool do_sample: Whether to generate by sampling
        :param float temperature: Only meaningful when do_sample is True
        :param int top_k: only sample from top_k
        :param float top_p: only sample from top_p token, nucles sample
        :param float repetition_penalty: How much to punish repeated tokens
        :param float length_penalty: Penalty for length, less than 1 encourages long sentences, greater than 1 encourages short plays
        :param int pad_token_id: After a sentence is generated, the generated content will be supplemented with pad_token_id
        """
        super().__init__()
        self.seq2seq_model = seq2seq_model
        self.restricter = restricter
        self.generator = SequenceGenerator(seq2seq_model.decoder, max_length=max_length, max_len_a=max_len_a,
                                           num_beams=num_beams,
                                           do_sample=do_sample,
                                           bos_token_id=bos_token_id,
                                           eos_token_id=eos_token_id,
                                           repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                           pad_token_id=pad_token_id,
                                           restricter=restricter)

    def forward(self, image_features, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None, first=None):
        """
        Transparently call the forward of seq2seq_model
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor tgt_tokens: bsz x max_len'
        :param torch.LongTensor src_seq_len: bsz
        :param torch.LongTensor tgt_seq_len: bsz
        :return:
        """
        return self.seq2seq_model(image_features, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first)

    def predict(self, image_features, src_tokens, src_seq_len=None, first=None):
        """
        Given the content of source, output the content of generate
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
        state = self.seq2seq_model.prepare_state(image_features, src_tokens, src_seq_len, first)
        result = self.generator.generate(state)
        return {'pred': result}