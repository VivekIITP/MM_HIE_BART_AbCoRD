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
        
        self.fusion_at_layer = [4]
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.image_encoder = list(self.vgg.children())[0]

        self.MAF_layer = MAF(dim_model=embed_dim,
                             dropout_rate=0.2)


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
        
        input_shape = input_ids.size()
        # print(input_shape)
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids)

        hidden_state = inputs_embeds + embed_pos
        hidden_state = self.layernorm_embedding(hidden_state)
        hidden_state = F.dropout(hidden_state, p=self.dropout, training=self.training)

        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        hidden_state = hidden_state.transpose(0, 1)
        
        # print("hidden states shape2:",hidden_states.shape)
        
        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # for encoder_layer in self.layers:
        for idx, encoder_layer in enumerate(self.layers):
            # ================================ Modifications ================================ #
            if idx in self.fusion_at_layer:
                # acoustic_input = self.acoustic_transformer(acoustic_input)[-1]
                # print(visual_input.shape)
                vgg_image_features = self.image_encoder(image_features)
                # print(vgg_image_features.shape)
                # # print('vgg: {}'.format(vgg_image_features.shape))
                # # print('1')
                # print("visual_context shape-2:",vgg_image_features.shape)
                vgg_image_features = vgg_image_features.permute(0, 2, 3, 1)
                # print("visual_context shape-1:",vgg_image_features.shape)
                vgg_image_features = vgg_image_features.reshape(
                    -1, 
                    vgg_image_features.size()[1]*vgg_image_features.size()[2], 
                    512
                    )
                hidden_state = self.MAF_layer(text_input=hidden_state,
                                               # acoustic_context=acoustic_input, 
                                               visual_context=vgg_image_features)
                # print("Image fused hidden states shape1:",hidden_states.shape)
                hidden_state = hidden_state.transpose(0, 1)
            # =============================================================================== #
            if output_hidden_states:
                # encoder_states.append(x)
                encoder_states.append(hidden_state)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                hidden_state, attn = None,None
            else:
                # x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)
                hidden_state, attn = encoder_layer(hidden_state, attention_mask, output_attentions=output_attentions)
                # print("Normal hidden states shape:",hidden_states.shape)
            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            # x = self.layer_norm(x)
            hidden_state = self.layer_norm(hidden_state)
        if output_hidden_states:
            # encoder_states.append(x)
            encoder_states.append(hidden_state)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        # x = x.transpose(0, 1)
        hidden_state = hidden_state.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [hidden_state, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=encoder_states, attentions=all_attentions)