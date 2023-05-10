import torch
import textdistance
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from transformers import BartTokenizer

import dataPipe, inference, loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#@title Loading trained model
# FOR LOADING MODEL FROM STATE DICT
# model.load_state_dict(torch.load('/content/drive/MyDrive/BART/Multimodal/Saved_models/dicts/MMbart_ImgContextAttn_VGG16.pth') #, map_location=torch.device('cpu')))
# FOR LOADING MODEL FROM MODEL FILE
model = torch.load('/content/drive/MyDrive/BART/Multimodal/Saved_models/MMbart_ImgConcat_fixSrcLen_Resnet152.pth')
model.to(device)

#@title Preparing Data
# data_bundle, tokenizer, mapping2id,mapping2targetid = get_data(f'/content/drive/MyDrive/Datasets/'+ds_name+'/json_text/',False) # FOR ONLY TEXT
data_bundle, tokenizer, mapping2id,mapping2targetid = get_MMdata(f'/content/drive/MyDrive/Datasets/'+ds_name,False) # FOR IMG FUSION USING MMBT
# data_bundle, tokenizer, mapping2id,mapping2targetid = get_MM_MAF_data(f'/content/drive/MyDrive/Datasets/'+ds_name,False) # FOR IMG FUSION USING MAF

# FOR FEW SHOT
# data_bundle, tokenizer, mapping2id,mapping2targetid = get_MMdata(f'/content/drive/MyDrive/BART/Datasets/Electronics/',False)
# raw_words_test = data_bundle.datasets['dev']['raw_words'].__dict__['content']

raw_words_test = data_bundle.datasets['test']['raw_words'].__dict__['content']
target_tokens_test = data_bundle.datasets['dev']['tgt_tokens'].__dict__['content']
# Uncomment below two line if not using images
target_images_test = data_bundle.datasets['dev']['image_features'].__dict__['content']
target_images_test = torch.Tensor(np.asarray([img.numpy() for img in target_images_test])).to(device)

data_bundle.delete_field(field_name='aspects')
data_bundle.delete_field(field_name='raw_words')

max_len = 20
max_len_a = 0.9 

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

idtarget2map=inv_map = {v: k for k, v in mapping2targetid.items()}

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())
vocab_size = len(tokenizer)

test_sentences = [" ".join(list1[1:]) for list1 in raw_words_test]
test_sentences_batch = []
target_tokens_batch = []
target_images_batch = []
prev = 0
for i in range(0,len(test_sentences),5):
  test_sentences_batch.append(test_sentences[prev:i+5])
  target_tokens_batch.append(target_tokens_test[prev:i+5])
  target_images_batch.append(target_images_test[prev:i+5])
  prev = i+5

#@title Generating results task wise
model.eval()
test_results_list = []

'''
Uncomment and comment the below code as per the model
'''
for sentences_batch,images_batch in zip(test_sentences_batch,target_images_batch):
# for sentences_batch in test_sentences_batch: # FOR ONLY TEXT
    # seg_token, seg_token_len = tokenize_sentence(sentences_batch,tokenizer,device) # FOR ONLY TEXT
    seg_token, seg_token_len = tokenize_sentence_mmconcat(sentences_batch,tokenizer,device)
    # seg_token, seg_token_len = tokenize_sentence_maf(sentences_batch,tokenizer,device) # FOR IMG FUSION USING MAF
    test_result = model.predict(images_batch, seg_token, seg_token_len)["pred"]
    # test_result = model.predict(seg_token, seg_token_len)["pred"]  # FOR ONLY TEXT
    test_results_list.append(test_result)

actual_triplets = []
predicted_triplets = []

for actual_batch,pred_batch,sent_batch in zip(target_tokens_batch,test_results_list,test_sentences_batch):
    # print(actual_batch)
    # print(pred_batch)
    # print(sent_batch)
    actual_triplets.extend(translateResult(sent_batch,actual_batch,idtarget2map,tokenizer))
    predicted_triplets.extend(translateResult(sent_batch,pred_batch,idtarget2map,tokenizer))


#@title Saving Results
# bart_test_results = {"actual_triplets":actual_triplets, "predicted_triplets":predicted_triplets}
# with open("/content/drive/MyDrive/BART/Saved_triplet_results/"+ds_name+"/Hie-Bart_SrcMaxLen100.json", "w") as outfile:
#     json.dump(bart_test_results, outfile)

#@title Type3 eval
pred_aspects = []
actual_aspects = []
pred_spans = []
actual_spans = []
pred_joint_spans = []
actual_joint_spans = []
pred_comp = []
actual_comp = []
consistent = 0
gen_noth = 0

for a_triplets,p_triplets in zip(actual_triplets,predicted_triplets):
  a_triplets.sort(key = lambda x: x[1])
  p_triplets.sort(key = lambda x: x[1])
  a_len = len(a_triplets)
  p_len = len(p_triplets)
  p_a = []
  a_a = []
  p_sp = ""
  a_sp = ""

  if(not p_len):
    # print("Generated nothing")
    gen_noth+=1
    continue

  if(a_len == p_len):
    consistent+=1
    for i in range(p_len):
      p_a.append(p_triplets[i][1])
      a_a.append(a_triplets[i][1])
      p_sp += p_triplets[i][0]
      a_sp += a_triplets[i][0]
      pred_comp.append(p_triplets[i][2])
      actual_comp.append(a_triplets[i][2])
      pred_spans.append(p_triplets[i][0])
      actual_spans.append(a_triplets[i][0])
    
    pred_aspects.append(tuple(p_a))
    actual_aspects.append(tuple(a_a))
    pred_joint_spans.append(p_sp)
    actual_joint_spans.append(a_sp)
  

  elif(a_len > p_len):
    for i in range(p_len):
      p_a.append(p_triplets[i][1])
      a_a.append(a_triplets[i][1])

    for i in range(p_len,a_len):
      a_a.append(a_triplets[i][1])
    
    p_a = list(dict.fromkeys(p_a))
    a_a = list(dict.fromkeys(a_a))
    p_len = len(p_a)
    
    # if not [value for value in p_a if value in a_a]:
    if(not (set(p_a).issubset(set(a_a)))):
    # if(p_a[-1] not in a_a):
        print('Not proper subset')
        print(p_triplets, a_triplets)
        print(p_a, a_a)
        for i in range(p_len):
          p_sp += p_triplets[i][0]
          a_sp += a_triplets[i][0]
          pred_comp.append(p_triplets[i][2])
          actual_comp.append(a_triplets[i][2])
          pred_spans.append(p_triplets[i][0])
          actual_spans.append(a_triplets[i][0])
    else:
        # print(p_triplets, a_triplets)
        # print(p_a, a_a)
        i=0
        j=0
        while(i < p_len):
            if(p_a[i] == a_a[j]):
              p_sp += p_triplets[i][0]
              a_sp += a_triplets[j][0]
              pred_comp.append(p_triplets[i][2])
              pred_spans.append(p_triplets[i][0])
              actual_comp.append(a_triplets[j][2])
              actual_spans.append(a_triplets[j][0])
              i+=1
              j+=1
            else:
              j+=1

    pred_aspects.append(tuple(p_a))
    actual_aspects.append(tuple(a_a))
    pred_joint_spans.append(p_sp)
    actual_joint_spans.append(a_sp)

  
  else:#(a_len<p_len)
    for i in range(a_len):
      p_a.append(p_triplets[i][1])
      a_a.append(a_triplets[i][1])

    for i in range(a_len,p_len):
      p_a.append(p_triplets[i][1])
    
    p_a = list(dict.fromkeys(p_a))
    a_a = list(dict.fromkeys(a_a))
    a_len = len(a_a)
    # if not [value for value in p_a if value in a_a]:
    if(not (set(a_a).issubset(set(p_a)))):
    # if(a_a[-1] not in p_a):
        print('Not proper subset')
        print(p_triplets, a_triplets)
        print(p_a, a_a)
        for i in range(a_len):
          p_sp += p_triplets[i][0]
          a_sp += a_triplets[i][0]
          pred_comp.append(p_triplets[i][2])
          actual_comp.append(a_triplets[i][2])
          pred_spans.append(p_triplets[i][0])
          actual_spans.append(a_triplets[i][0])
    else:
        i=0
        j=0
        while(j < a_len):
            if(p_a[i] == a_a[j]):
              p_sp += p_triplets[i][0]
              a_sp += a_triplets[j][0]
              pred_comp.append(p_triplets[i][2])
              pred_spans.append(p_triplets[i][0])
              actual_comp.append(a_triplets[j][2])
              actual_spans.append(a_triplets[j][0])
              i+=1
              j+=1
            else:
              i+=1
    pred_aspects.append(tuple(p_a))
    actual_aspects.append(tuple(a_a))
    pred_joint_spans.append(p_sp)
    actual_joint_spans.append(a_sp)

print()
print()
print("Consistent len {} %, {} instances generated nothing".format(round(100*consistent/len(actual_triplets),2),gen_noth))
print()
print()

#Multilabel Aspect Evaluation
print('Multilabel Aspect Evaluation')
mlb = MultiLabelBinarizer()
act_asp_labels = mlb.fit_transform(actual_aspects)
pred_asp_labels = mlb.transform(pred_aspects)
print("Classes:",list(mlb.classes_))
print(classification_report(act_asp_labels, pred_asp_labels,target_names=list(mlb.classes_)))
print()
print()

#title Span Evaluation Separate(string similarity)
print('Span Evaluation Separate(string similarity)')
hamming_distance = []
hamming_similarity = []
jaccard_index = []
levenshtein_similarity = []
ratcliff_obershelp_similarity = []
for st1,st2 in zip(actual_spans,pred_spans):

  hamming_distance.append(textdistance.hamming.normalized_distance(st1,st2))
  hamming_similarity.append(textdistance.hamming.normalized_similarity(st1,st2))
  jaccard_index.append(textdistance.jaccard(st1,st2))
  levenshtein_similarity.append(textdistance.levenshtein.normalized_similarity(st1,st2))
  ratcliff_obershelp_similarity.append( textdistance.ratcliff_obershelp(st1,st2))

total = len(actual_spans)
avg_hamming_distance = sum(hamming_distance)/total
avg_hamming_similarity = sum(hamming_similarity)/total
avg_jaccard_index = sum(jaccard_index)/total
avg_levenshtein_similarity = sum(levenshtein_similarity)/total
avg_ratcliff_obershelp_similarity = sum(ratcliff_obershelp_similarity)/total

print("Avg_hamming_distance               ",avg_hamming_distance)
print("Avg_hamming_similarity             ",avg_hamming_similarity)
print("Avg_jaccard_index                  ",avg_jaccard_index)
print("Avg_levenshtein_similarity         ",avg_levenshtein_similarity)
print("Avg_ratcliff_obershelp_similarity  ",avg_ratcliff_obershelp_similarity)

print()
#title Span Evaluation Joint (string similarity)
print('Span Evaluation Joint(string similarity)')
hamming_distance = []
hamming_similarity = []
jaccard_index = []
levenshtein_similarity = []
ratcliff_obershelp_similarity = []
for st1,st2 in zip(actual_joint_spans,pred_joint_spans):

  hamming_distance.append(textdistance.hamming.normalized_distance(st1,st2))
  hamming_similarity.append(textdistance.hamming.normalized_similarity(st1,st2))
  jaccard_index.append(textdistance.jaccard(st1,st2))
  levenshtein_similarity.append(textdistance.levenshtein.normalized_similarity(st1,st2))
  ratcliff_obershelp_similarity.append( textdistance.ratcliff_obershelp(st1,st2))

total = len(actual_joint_spans)
avg_hamming_distance = sum(hamming_distance)/total
avg_hamming_similarity = sum(hamming_similarity)/total
avg_jaccard_index = sum(jaccard_index)/total
avg_levenshtein_similarity = sum(levenshtein_similarity)/total
avg_ratcliff_obershelp_similarity = sum(ratcliff_obershelp_similarity)/total

print("Avg_hamming_distance               ",avg_hamming_distance)
print("Avg_hamming_similarity             ",avg_hamming_similarity)
print("Avg_jaccard_index                  ",avg_jaccard_index)
print("Avg_levenshtein_similarity         ",avg_levenshtein_similarity)
print("Avg_ratcliff_obershelp_similarity  ",avg_ratcliff_obershelp_similarity)

#title Complaint Identification
print()
print()
print('Complaint Identification')
print(classification_report(actual_comp, pred_comp))
print(accuracy_score(actual_comp, pred_comp))