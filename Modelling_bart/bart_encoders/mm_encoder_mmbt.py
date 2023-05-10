#@title modeling MMBart(ImgConcatenation)

class MultiModalBartEncoder(nn.Module):
# class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.
    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                2,# config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

        self.num_image_embeds = 1

        self.img_embeddings = ImageBartEmbeddings(
                                                 self.num_image_embeds,
                                              self.embed_tokens, 
                                          self.embed_positions,
                                      self.layernorm_embedding
                                     )


    def forward(
            self, input_ids, image_features, 
            attention_mask=None, output_attentions=False, 
            output_hidden_states=False, return_dict=False
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            BaseModelOutput or Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (tuple(torch.FloatTensor)): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *output_hidden_states:* is True.
                - **all_attentions** (tuple(torch.FloatTensor)): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # print(input_ids.shape)
        trim_input_ids = input_ids[:, :-2]
        # print(trim_input_ids.shape)
        img_embed = self.img_embeddings(image_features)
        # print("Img embed shape:",img_embed.shape)
        txt_embed = self.embed_tokens(trim_input_ids)
        # print("txt_embed shape:",txt_embed.shape)
        txt_embed = txt_embed * self.embed_scale
        embed_pos = self.embed_positions(trim_input_ids)
        txt_embed = txt_embed + embed_pos
        txt_embed = self.layernorm_embedding(txt_embed)
        txt_embed = F.dropout(txt_embed, p=self.dropout, training=self.training)
        hidden_state = joint_input_embedding = torch.cat([txt_embed, img_embed], 1)
        # print(hidden_state.shape)
        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        hidden_state = hidden_state.transpose(0, 1)
        
        # print("hidden states shape2:",hidden_states.shape)
        
        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # for encoder_layer in self.layers:
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(hidden_state)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                hidden_state, attn = encoder_layer(hidden_state, attention_mask, output_attentions=output_attentions)
                # print("Normal hidden states shape:",hidden_states.shape)
            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            hidden_state = self.layer_norm(hidden_state)
        if output_hidden_states:
            encoder_states.append(hidden_state)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        hidden_state = hidden_state.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [hidden_state, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=encoder_states, attentions=all_attentions)

