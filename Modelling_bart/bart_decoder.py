"""PyTorch BART model, ported from the fairseq repo."""

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
]


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def _prepare_bart_decoder_inputs(
        config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    if decoder_padding_mask is not None and decoder_padding_mask.shape[1] > 1:
        # never mask leading token, even if it is pad
        decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]
    tmp = fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp.to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
    return decoder_input_ids, decoder_padding_mask, causal_mask


class PretrainedBartModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
            self,
            x,
            encoder_hidden_states,
            encoder_attn_mask=None,
            layer_state=None,
            causal_mask=None,
            decoder_padding_mask=None,
            output_attentions=False,
    ):
        residual = x
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention
        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        # self.do_blenderbot_90_layernorm = config.do_blenderbot_90_layernorm  # layernorm variant
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        # if config.static_position_embeddings:
        #     self.embed_positions = SinusoidalPositionalEmbedding(
        #         config.max_position_embeddings, config.d_model, config.pad_token_id
        #     )
        # else:
        self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
                2,# config.extra_pos_embeddings
            )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None
        self.config = config

    def set_position_embedding(self, special_tag_start_id, tag_first=True):
        if tag_first:
            embed_positions = DecoderLearnedPositionalEmbedding(
                self.config.max_position_embeddings,
                self.config.d_model,
                self.padding_idx,
                2,# self.config.extra_pos_embeddings,
                special_tag_start_id
            )
        else:
            embed_positions = DecoderLearnedPositionalEmbedding2(
                self.config.max_position_embeddings,
                self.config.d_model,
                self.padding_idx,
                2,# self.config.extra_pos_embeddings,
                special_tag_start_id
            )

        embed_positions.weight.data = self.embed_positions.weight.data
        self.embed_positions = embed_positions

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask,
            decoder_padding_mask,
            decoder_causal_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            use_pos_cache=False,
            **unused,):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            past_key_values (dict or None): dictionary used for storing state during generation
        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states
                - attentions
        """

        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")

        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)

        # embed positions
        positions = self.embed_positions(input_ids, use_cache=use_pos_cache)

        if use_pos_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]

        x = self.embed_tokens(input_ids) * self.embed_scale
        # if self.do_blenderbot_90_layernorm:
        #     x = self.layernorm_embedding(x)
        #     x += positions
        # else:
        x += positions
        x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = past_key_values[idx] if past_key_values is not None else None

            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
            )

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if output_attentions:
                all_self_attns += (layer_self_attn,)

        if self.layer_norm:  # if config.add_final_layer_norm (mBART)
            x = self.layer_norm(x)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class DecoderLearnedPositionalEmbedding(nn.Embedding):
    """
    主要修改是，position的是循环的
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset,
                 special_tag_start_id=None):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        self.special_tag_start_id = special_tag_start_id  # 这个id之后的词是特殊词汇
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if self.special_tag_start_id is None or input_ids.size(1)<2:
            bsz, seq_len = input_ids.shape[:2]
            if use_cache:
                positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
            else:
                # starts at 0, ends at 1-seq_len
                positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        else:
            # 实现的是每个位置重新开始position
            """
                大概意思是，假设input_ids中假设大于4是特殊符号，那么输入是
                [[2, 4, 1, 2, 3, 5, 1],
                 [2, 5, 3, 3, 0, 0, 0]]时，输出为
                [[0, 1, 2, 3, 4, 1, 2],
                 [0, 1, 2, 3, 4, 5, 6] 每个大于4的位置都会重置
            """
            _input_ids = input_ids[:, 1:]
            bsz, seq_len = _input_ids.shape[:2]
            special_tag_mask = _input_ids.ge(self.special_tag_start_id)  # bsz x max_len
            if special_tag_mask.sum()>0:
                num_masks = special_tag_mask.cumsum(dim=1).max()  # 表示最长的
                arange_indices = torch.arange(seq_len).to(_input_ids).expand_as(_input_ids)  # bsz x max_len
                special_tag_indice = arange_indices.masked_select(special_tag_mask)  # a vector只包含所有的special的indice
                indices = torch.arange(num_masks).to(_input_ids)[None].repeat(bsz, 1)  # bsz x mask_len
                mask = indices.lt(special_tag_mask.sum(dim=1, keepdim=True))
                indices = indices.masked_scatter(mask, special_tag_indice)
                _, inverted_indices = special_tag_mask.cumsum(dim=-1).unique(return_inverse=True)

                inverted_indices = inverted_indices - inverted_indices[:, :1]
                inverted_indices = inverted_indices.masked_fill(inverted_indices.ge(indices.size(1)), max(indices.size(1)-1, 0))
                positions = indices.gather(index=inverted_indices, dim=1)
                positions = (arange_indices - positions) + 1
            else:
                positions = torch.arange(seq_len+1, dtype=torch.long, device=self.weight.device)[None]

            if use_cache:
                positions = positions[:, -1:]
            else:
                positions = torch.cat([input_ids.new_zeros(bsz, 1), positions], dim=1)

        return super().forward(positions + self.offset)


class DecoderLearnedPositionalEmbedding2(nn.Embedding):
    """
    主要修改是，position的是循环的, 和上面的区别是tag所在的位置不同
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset,
                 special_tag_start_id=None):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        self.special_tag_start_id = special_tag_start_id  # 这个id之后的词是特殊词汇
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if self.special_tag_start_id is None or input_ids.size(1)<2:
            bsz, seq_len = input_ids.shape[:2]
            if use_cache:
                positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
            else:
                # starts at 0, ends at 1-seq_len
                positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        else:
            # 实现的是每个位置重新开始position
            """
                大概意思是，假设input_ids中假设大于4是特殊符号，那么输入是
                [[2, 1, 2, 3, 4, 1, 5],
                 [2, 3, 3, 5, 0, 0, 0]]时，输出为
                [[0, 1, 2, 3, 4, 1, 2],
                 [0, 1, 2, 3, 4, 5, 6] 每个大于4的位置都会重置
            """
            _input_ids = input_ids[:, 1:]  # 把sos去掉
            bsz, seq_len = _input_ids.shape[:2]
            special_tag_mask = _input_ids.ge(self.special_tag_start_id)  # bsz x max_len
            if special_tag_mask.sum()>0:
                num_masks = special_tag_mask.cumsum(dim=1)  # 表示最长的
                num_masks_value = num_masks.max()
                arange_indices = torch.arange(seq_len).to(_input_ids).expand_as(_input_ids)  # bsz x max_len

                special_tag_indice = arange_indices.masked_select(special_tag_mask)  # a vector只包含所有的special的indice
                indices = torch.arange(num_masks_value).to(_input_ids)[None].repeat(bsz, 1)  # bsz x mask_len
                mask = indices.lt(special_tag_mask.sum(dim=-1, keepdim=True))
                special_tag_indice = indices.masked_scatter(mask, special_tag_indice)

                indices = torch.cat([special_tag_indice.new_zeros(bsz, 1), special_tag_indice[:, :-1] + 1], dim=1)
                _, inverted_indices = special_tag_mask.flip(dims=[1]).cumsum(dim=-1).flip(dims=[1]).unique(
                    return_inverse=True)
                values = inverted_indices[:, 0]  # bsz
                inverted_indices = values[:, None] - inverted_indices
                inverted_indices = inverted_indices.masked_fill(inverted_indices.ge(indices.size(1)), indices.size(1)-1)

                positions = indices.gather(index=inverted_indices, dim=1)
                positions = arange_indices - positions + 1
            else:
                positions = torch.arange(seq_len+1, dtype=torch.long, device=self.weight.device)[None]

        if use_cache:
            positions = positions[:, -1:]
        else:
            positions = torch.cat([input_ids.new_zeros(bsz, 1), positions], dim=1)

        return super().forward(positions + self.offset)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


# Public API
def _get_shape(t):
    return getattr(t, "shape", None)

class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
        The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0: dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # This line breaks for odd n_pos
        out[:, dim // 2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)
