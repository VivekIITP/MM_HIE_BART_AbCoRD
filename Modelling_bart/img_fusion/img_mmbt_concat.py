class ImageEncoder(nn.Module):
    def __init__(self,model_name, num_image_embeds):
        super(ImageEncoder, self).__init__()
        hidden_sz = 768

        if(model_name == "vgg16"):
            img_hidden_sz = 512
            vgg16 = torchvision.models.vgg16(pretrained=True)
            vgg16_modules = list(vgg16.children())[:-2]
            self.model = nn.Sequential(*vgg16_modules)
        elif(model_name == "vgg19"):
            img_hidden_sz = 512
            vgg19 = torchvision.models.vgg19(pretrained=True)
            vgg19_modules = list(vgg19.children())[:-2]
            self.model = nn.Sequential(*vgg19_modules)
        elif(model_name == "resnet152"):
            img_hidden_sz = 2048
            resnet152 = torchvision.models.resnet152(pretrained=True)
            resnet152_modules = list(resnet152.children())[:-2]
            self.model = nn.Sequential(*resnet152_modules)
        elif(model_name == "resneXt_101_32x8d"):
            img_hidden_sz = 2048
            resneXt_101_32x8d = torchvision.models.resnext101_32x8d(pretrained=True)
            resneXt_101_32x8d_modules = list(resneXt_101_32x8d.children())[:-2]
            self.model = nn.Sequential(*resneXt_101_32x8d_modules)
        elif(model_name == "densenet161"):
            img_hidden_sz = 2208
            densenet161 = torchvision.models.densenet161(pretrained=True)
            densenet161_modules = list(densenet161.children())[:-1]
            self.model = nn.Sequential(*densenet161_modules)
        else:
            print("Invalid model")

        pool_func = (
            nn.AdaptiveAvgPool2d)
            # if img_embed_pool_type == "avg"
            # else nn.AdaptiveMaxPool2d)

        self.pool = pool_func((num_image_embeds, 1))
        self.linearize_out = nn.Linear(img_hidden_sz, hidden_sz)

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048 ->BxNx768 for resnet models
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        out = self.linearize_out(out)
        return out  # BxNx768

class ImageBartEmbeddings(nn.Module):
    def __init__(self, num_image_embeds , embed_tokens, embed_positions, layernorm_embedding):
        super(ImageBartEmbeddings, self).__init__()

        model_name = IMG_MODEL
        dropout = 0.1
        self.img_encoder = ImageEncoder(model_name, num_image_embeds)
        self.embed_positions = embed_positions
        self.embed_tokens = embed_tokens
        self.layernorm_embedding = layernorm_embedding
        self.dropout = nn.Dropout(p = dropout)
        self.num_image_embeds = num_image_embeds  # 1
        self.tokenizer = tokenizer

    def forward(self, input_images):
        imgs_embeddings = self.img_encoder(input_images)  # BxNx3x224x224 -> BxNx768 for resnet models
        bsz = imgs_embeddings.size(0)
        seq_length = self.num_image_embeds + 1  # +1 SEP Token

        sep_id = torch.LongTensor([self.tokenizer.convert_tokens_to_ids("</s>")]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.embed_tokens(sep_id)

        token_embeddings = torch.cat(
            [imgs_embeddings, sep_token_embeds], dim=1
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        embed_positions = self.embed_positions(position_ids)
        embeddings = token_embeddings + embed_positions
        embeddings = self.layernorm_embedding(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings