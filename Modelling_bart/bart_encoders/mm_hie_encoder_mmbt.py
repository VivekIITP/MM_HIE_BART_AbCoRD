class MultiModalBartEncoder(nn.Module):
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
        self.embed_dim = embed_dim    # For Hie
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
        self.quant_noise = None

        self.layer_wise_attention = False  #config.layer_wise_attention

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.num_layers = len(self.layers)

        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

        self.device = device #torch.device('cuda')

        self.num_image_embeds = 1

        self.img_embeddings = ImageBartEmbeddings( self.num_image_embeds,
                                                self.embed_tokens, 
                                            self.embed_positions,
                                        self.layernorm_embedding
                                     )
    
    def segment_sent(self, tokens):    # For Hie
        # [B, T]  <cls>:50261 , PAD:1
        segment_sent = tokens.clone()
        segment_flag = tokens.eq(50261)
        
        segment_list = []
        zero_segments = torch.zeros([1]).to(self.device).half()
        one_segments = torch.ones([1]).to(self.device).half()
        for i in range(segment_flag.shape[0]):
            cnt = -1
            segments_mini_list = []
            
            for j in range(segment_flag.shape[1]):    # Every Batch
                if segment_flag[i][j] == True:
                    cnt += 1
                
                if segment_sent[i][j] == self.padding_idx:
                    segments_mini_list.append(zero_segments)
                elif cnt % 2 == 0:
                    segments_mini_list.append(zero_segments)
                elif cnt % 2 != 0:
                    segments_mini_list.append(one_segments)
                    
            segments = torch.cat(segments_mini_list)
            segment_list.append(segments)
            
        segments_emb = torch.stack(segment_list, dim=0)
        segment_sent = segments_emb.unsqueeze(-1).repeat(1, 1, self.embed_dim)  # [B, T] -> [B, T, C]
        return segment_sent

    def forward_embedding(self, input_ids):
        # embed tokens, img feat and positions
        x = embed = self.embed_scale * self.embed_tokens(input_ids)
        # x = embed = self.embed_scale * joint_input_embedding
        #if self.segmenting:
        #x = embed + self.segment_sent(input_ids)  # For Hie
        
        if self.embed_positions is not None:
            x = embed + self.embed_positions(input_ids)
            
        #x = x + self.segment_sent(input_ids)  # For Hie
        
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward_sent_embedding(self, input_ids):  # For Hie
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(input_ids)
        
        x = embed + self.segment_sent(input_ids)
        
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self, 
            input_ids, 
            image_features,
            attention_mask=None, 
            output_attentions=False, 
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
        segments = input_ids.clone()     # For Hie
        segments = segments.eq(50261)

        # print(input_ids.shape)
        trim_input_ids = input_ids[:, :-2]
        # print(trim_input_ids.shape)
        img_embed = self.img_embeddings(image_features)
        # print("Img embed shape:",img_embed.shape)
        txt_embed, _ = self.forward_embedding(trim_input_ids)  # encoder_embedding -> inputs_embeds

        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        #sent_x, sent_encoder_embedding = self.forward_sent_embedding(input_ids)  # For Hie
        
        x = joint_input_embedding = torch.cat([txt_embed, img_embed], 1)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        #sent_x = sent_x.transpose(0, 1)  # For Hie

        encoder_states = [] if output_hidden_states else None

        # all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
                #encoder_states.append(sent_x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                # x, attn =
                x = encoder_layer(x, attention_mask, output_attentions=output_attentions, 
                segments=segments, src=input_ids)

            # if output_attentions:
            #     all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if not return_dict:
            # return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
            return tuple(v for v in [x, encoder_states] if v is not None)
            
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=None)