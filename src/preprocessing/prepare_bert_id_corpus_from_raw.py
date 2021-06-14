from transformers import AutoTokenizer
import torch
import random
import sys

input_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/"
input_file = input_dir + "book_both_all"
output_dir = "/iesl/canvas/hschang/language_modeling/Multi-facet_softmax/data/processed/book_bert/"
output_train_file = output_dir + "book_org_train_mid_cased.pt"
output_val_file = output_dir + "book_org_val_mid_cased.pt"
#input_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/"
#input_file = input_dir + "wiki2016_both.txt"
#output_dir = "/iesl/canvas/hschang/language_modeling/Multi-facet_softmax/data/processed/wiki2016_bert/"
#output_train_file = output_dir + "wiki2016_org_train_mid.pt"
#output_val_file = output_dir + "wiki2016_org_val_mid.pt"

#output_train_file = output_dir + "wiki2016_org_train_small.pt"
#output_val_file = output_dir + "wiki2016_org_val_small.pt"
#output_train_file = output_dir + "wiki2016_org_train_ms.pt"
#output_val_file = output_dir + "wiki2016_org_val_ms.pt"

#double_sent = True
double_sent = False

training_ratio = 0.99

#max_line_num = 1000000000000
#max_line_num = 100000
max_line_num = 10000000
#max_line_num = 20000000
#max_line_num = 2000000

max_sent_len = 256

output_arr = []

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

i=0
with open(input_file) as f_in:
    for line in f_in:
        raw_text = line.rstrip().split('\t')[0]
        if len(raw_text) == 0:
            continue
        i+=1
        indexed_tokens = tokenizer.encode(raw_text)
        if double_sent:
            if i % 2 == 0:
                last_indexed_tokens = indexed_tokens
                continue
            else:
                indexed_tokens = last_indexed_tokens + indexed_tokens[1:]
        output_arr.append(indexed_tokens)
        if i % 100000 == 0:
            print(i)
            sys.stdout.flush()
        if i > max_line_num:
            break

idx_shuffled = list(range(len(output_arr)))
random.shuffle(idx_shuffled)
training_size = int(len(output_arr)*training_ratio)

def save_to_tensor(output_arr, output_file_name):
    data_size = len(output_arr)
    output_tensor = torch.zeros((data_size,max_sent_len),dtype = torch.int16)

    for i in range(data_size):
        sent_len = min(max_sent_len, len(output_arr[i]))
        output_tensor[i,:sent_len] = torch.tensor(output_arr[i][:sent_len],dtype = torch.int16)

    torch.save(output_tensor, output_file_name)

save_to_tensor([output_arr[i] for i in idx_shuffled[:training_size]], output_train_file)
save_to_tensor([output_arr[i] for i in idx_shuffled[training_size:]], output_val_file)
