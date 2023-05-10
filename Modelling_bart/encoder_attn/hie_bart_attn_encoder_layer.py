join_pt = 4

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim,
            num_heads,
            join_head=0,
            dropout=0.0,
            bias=True,
            encoder_decoder_attention=False,  # otherwise self_attention
            
            self_attention=False,
            kdim=None,
            vdim=None,
            add_bias_kv=False,
            add_zero_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.half_num_heads = num_heads // 2 # For Hie
        self.half_head_dim = embed_dim // num_heads
        self.half_scaling = self.half_head_dim ** -0.5
        self.heads_attn = None # For Hie

        self.join_head = join_head # For Hie

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.enable_torch_version = False

        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
            attn_mask: Optional[Tensor] = None,
            output_attentions=False,

            hie_self_attn: bool = False,    #  For Hie
            sent_key_padding_mask: Optional[Tensor] = None,  # For Hie
            word_attn: Optional[Tensor] = None,    # For Hie
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        """Input shape: Time(SeqLen) x Batch x Channel"""
        
        static_kv: bool = self.encoder_decoder_attention

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute key and value if they are static
                key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        
        if self.self_attention and not hie_self_attn:  # For Hie
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)
        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)
        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (
            bsz,
            src_len,
        )

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        
        # For Hie
        heads_attn = attn_output.clone()

        if hie_self_attn:
            adj_head = self.half_num_heads + self.join_head
            attn_output = torch.cat([self.heads_attn[:bsz * adj_head, :, :], heads_attn[bsz * adj_head:, :, :]], dim=0)  # word_sent adj
            
        self.heads_attn = heads_attn

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask


class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(
            self.embed_dim, 
            config.encoder_attention_heads, 
            join_head = join_pt, 
            dropout=config.attention_dropout,
            self_attention=True
            )
        self.self_attn.enable_torch_version = False    # For Hie

        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, output_attentions=False, 
            attn_mask: Optional[Tensor] = None, segments = None, src = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        
        # x, attn_weights = self.self_attn(
        #     query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions
        # ) # THIS IS REPLACED BY WORD AND SENTENCE LEVEL ATTENTIONS BELOW

        ### For Hie
        x_origin = x.clone()

        ### Word Level Attention
        sent_x, attn_weights = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            output_attentions=True,
        )

        ### x             [len, b, h]
        ### sent_x        [len, b, h]
        ### attn_weights  [head, b, len, len]
        ### segments      [b, len]
        sentence_x = []
        sentences_len = []
        for b_idx in range(segments.shape[0]):
            segments_idx = torch.nonzero(segments[b_idx])
            sent_len = segments_idx.shape[0]
            tmp_sentence_x = []
            for w_idx in range(sent_len):
                front = segments_idx[w_idx]
                if w_idx == sent_len - 1:
                    back = -1
                else:
                    back = segments_idx[w_idx+1]
                
                sent_inc = x[front:back, b_idx, :]
                
                #max_sent_vec = torch.max(sent_inc, 0, keepdim=True)   # [valuse, indices]    # max pooling
                #tmp_sentence_x.append(max_sent_vec[0])
                max_sent_vec = torch.mean(sent_inc, 0, keepdim=True)   # [valuse, indices]  # avg pooling
                tmp_sentence_x.append(max_sent_vec)
                
            if len(tmp_sentence_x) == 0:
                continue
            elif len(tmp_sentence_x) == 1:
                tmp_sentence_x = tmp_sentence_x[0]
            else:
                tmp_sentence_x = torch.cat(tmp_sentence_x)
                
            if not len(tmp_sentence_x) == 0:
                sentence_x.append(tmp_sentence_x)
                sentences_len.append(sentence_x[b_idx].shape[0])

        if len(tmp_sentence_x) == 0:
            sentences = x
            #sentences = sent_x
            sent_padding_mask = encoder_padding_mask
        else:
            sentences = torch.nn.utils.rnn.pad_sequence(sentence_x, batch_first=True)    # [B, s_len, hidden]
            sent_padding_mask = torch.ge(-torch.abs(sentences[:, :, 0]), 0.0)
            sentences = sentences.transpose(0, 1)                                        # [s_len, B, hidden]
        sent_attn_mask = None  # tmp

        if sent_attn_mask is not None:
            sent_attn_mask = sent_attn_mask.masked_fill(sent_attn_mask.to(torch.bool), -1e8)
            
        word_attn = self.self_attn.heads_attn

        ### Sentence Level Attention
        x, sent_attn_weights = self.self_attn(
            query=x_origin,   # x_origin or x or sent_x
            key=sentences,
            key_padding_mask=sent_padding_mask,   # encoder_padding_mask
            attn_mask=sent_attn_mask,
            output_attentions=True,
            hie_self_attn=True,
            word_attn=word_attn,
        )


        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

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
        
        if torch.isinf(x).any() or torch.isnan(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x  #, attn_weights
