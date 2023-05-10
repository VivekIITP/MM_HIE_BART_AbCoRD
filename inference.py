# @title For inference translateResult
from itertools import chain

def translateResult(sentences,results,idtarget2map,tokenizer):
    converted_sentences=[]
    for i in sentences:
        lst = []
        for word in i.split(" "):
            bpes = tokenizer.tokenize(word, add_prefix_space=True)
            lst.extend(bpes)
        converted_sentences.append(" ".join(lst))
    output_mold = []
    # len_lst = []
    cap = len(idtarget2map)+2
    def translateResultChunk(block,sentence):
        output = []
        prev = None
        lst = sentence.split(" ")
        for i in block:
              if i<cap:
                    output.append(idtarget2map[i-2])
                    continue
              if prev is not None:
                    if(prev < 0 or i-cap <= prev):
                        output.append("NO_CAUSE")
                    else:
                        chunk = lst[prev:i-cap]
                        text = ""
                        for i in chunk:
                            if i[0]=="Ġ":
                                text =text +" "+i[1:]
                            else:
                                text =text +i
                        output.append(text.strip())
                        prev = None
              else:
                    prev = i-cap-1
        return output
    for i in range(len(converted_sentences)):
        blocks=[]
        category_check = False
        cur_pair = []
        for o in results[i]:
            k = int(o)
            if k==0 or k==1: # start/end ignore
                        continue
                    
            if k<cap:    # category/nature encoding
                  cur_pair.append(k)

                  if not category_check:
                      category_check = True
                  else:
                      if len(cur_pair) == 4:  # and cur_pair[0] <= cur_pair[1]):
                                blocks.append(translateResultChunk(cur_pair,"NO_CAUSE "+converted_sentences[i]).copy())
                      cur_pair = []
                      category_check = False
            else:
                  cur_pair.append(k)
        output_mold.append(blocks.copy())
    return output_mold

    #@title For only text tokenize_sentence
def tokenize_sentence(sentences,tokenizer,device = "cpu"):
    output_mold = []
    len_lst = []
    src_max_len = SOURCE_MAX_LEN - 1
    for i in range (0,len(sentences)):
        added_sentence = "<<no_cause>> "+sentences[i]
        raw_words = added_sentence.split(" ")
        word_bpes = [[tokenizer.bos_token_id]]
        for word in raw_words:
            bpes = tokenizer.tokenize(word, add_prefix_space=True)
            bpes = tokenizer.convert_tokens_to_ids(bpes)
            word_bpes.append(bpes)

        src_input_ids = list(chain(*word_bpes))
        curr_len = len(src_input_ids)
            
        if curr_len < src_max_len:
              pad_no = src_max_len - curr_len
              for i in range(pad_no):
                  src_input_ids.append(tokenizer.pad_token_id)
        else: # >=
              src_input_ids = src_input_ids[:src_max_len]

        src_input_ids.append(tokenizer.eos_token_id)        
        output = src_input_ids
        output_mold.append(output)
    max_len = max(len(x) for x in output_mold)
    mold_np = np.ones([len(sentences),max_len])
    for i in range (0,len(sentences)):
        raw_words = output_mold[i]
        len_lst.append(len(raw_words))
        mold_np[i,:len_lst[-1]]=raw_words
    seg_token = torch.LongTensor(mold_np).to(device)
    seg_token_len = torch.LongTensor(len_lst).to(device)
    return seg_token , seg_token_len

#@title For ImgCONCAT tokenize_sentence
def tokenize_sentence_mmconcat(sentences,tokenizer,device = "cpu"):
    output_mold = []
    len_lst = []
    src_max_len = SOURCE_MAX_LEN - 3 # for sep_id, img_id, eos_id
    for i in range (0,len(sentences)):
        added_sentence = "<<no_cause>> "+sentences[i]
        raw_words = added_sentence.split(" ")
        word_bpes = [[tokenizer.bos_token_id]]
        for word in raw_words:
            bpes = tokenizer.tokenize(word, add_prefix_space=True)
            bpes = tokenizer.convert_tokens_to_ids(bpes)
            word_bpes.append(bpes)

        src_input_ids = list(chain(*word_bpes))
        curr_len = len(src_input_ids)
            
        if curr_len < src_max_len:
          pad_no = src_max_len - curr_len
          for i in range(pad_no):
              src_input_ids.append(tokenizer.pad_token_id)
        else: # >=
          src_input_ids = src_input_ids[:src_max_len]

        src_input_ids.append(tokenizer.sep_token_id)
        src_input_ids.append(tokenizer.convert_tokens_to_ids("<<img_feat>>"))
        src_input_ids.append(tokenizer.eos_token_id)
        output_mold.append(src_input_ids)
    
    max_len = max(len(x) for x in output_mold)
    mold_np = np.ones([len(sentences),max_len])
    for i in range (0,len(sentences)):
        raw_words = output_mold[i]
        len_lst.append(len(raw_words))
        mold_np[i,:len_lst[-1]]=raw_words
    seg_token = torch.LongTensor(mold_np).to(device)
    seg_token_len = torch.LongTensor(len_lst).to(device)
    return seg_token , seg_token_len

#@title For ImgContextAttn tokenize_sentence
def tokenize_sentence_maf(sentences,tokenizer,device = "cpu"):
    output_mold = []
    len_lst = []
    src_max_len = SOURCE_MAX_LEN - 1 # for sos
    for i in range (0,len(sentences)):
        added_sentence = "<<no_cause>> "+sentences[i]
        raw_words = added_sentence.split(" ")
        word_bpes = [[tokenizer.bos_token_id]]
        for word in raw_words:
                  bpes = tokenizer.tokenize(word, add_prefix_space=True)
                  bpes = tokenizer.convert_tokens_to_ids(bpes)
                  word_bpes.append(bpes)

        src_input_ids = list(chain(*word_bpes))
        curr_len = len(src_input_ids)
            
        if curr_len < src_max_len:
              pad_no = src_max_len - curr_len
              for i in range(pad_no):
                  src_input_ids.append(tokenizer.pad_token_id)
        else: # >=
              src_input_ids = src_input_ids[:src_max_len]

        src_input_ids.append(tokenizer.eos_token_id)
        output_mold.append(src_input_ids)
    
    max_len = max(len(x) for x in output_mold)
    mold_np = np.ones([len(sentences),max_len])
    for i in range (0,len(sentences)):
        raw_words = output_mold[i]
        len_lst.append(len(raw_words))
        mold_np[i,:len_lst[-1]]=raw_words
    seg_token = torch.LongTensor(mold_np).to(device)
    seg_token_len = torch.LongTensor(len_lst).to(device)
    return seg_token , seg_token_len