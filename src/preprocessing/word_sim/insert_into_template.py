import os
import sys
sys.path.insert(0, sys.path[0]+'/../..')

import torch
from gpt2_model.tokenization_gpt2 import GPT2Tokenizer

template_nouns = ["I love the $ARG1 and the $ARG2, and my favorite is the",
                  "Yesterday, a man encounters the $ARG1 and the $ARG2. Today, he again saw the",
                  "There are the $ARG1 and the $ARG2 in front of a woman, and she decide to pursue the",
                  "If you can choose the $ARG1 or the $ARG2, would you choose the"]

#input_folder = "data/processed/similarity_MEN/split"
#output_text_dir = "data/processed/similarity_MEN/text_more"
#output_tensor_dir = "data/processed/similarity_MEN/tensor_more"

input_folder = "data/processed/similarity_hyper/split"
output_text_dir = "data/processed/similarity_hyper/text_more"
output_tensor_dir = "data/processed/similarity_hyper/tensor_more"

if not os.path.exists(output_text_dir):
    os.makedirs(output_text_dir)
if not os.path.exists(output_tensor_dir):
    os.makedirs(output_tensor_dir)

tokenizer_GPT2 = GPT2Tokenizer.from_pretrained('gpt2')

def get_last_id(template_arr):
    template_last_id_arr = []
    for sent in template_arr:
        sent_idx = tokenizer_GPT2.encode(sent)
        template_last_id_arr.append(sent_idx[-1])
        #print(sent, sent_idx)
    return template_last_id_arr

tempaate_nouns_last_arr = get_last_id(template_nouns)

def load_input(input_path):
    input_data = []
    with open(input_path) as f_in:
        for line in f_in:
            w1, w2 = line.rstrip().split()
            input_data.append([w1, w2])
    return input_data

def create_data(template, w1, w4):
    sent_arr = []    
    assert template.split()[-1] != w1
    assert template.split()[-1] != w4
    sent_arr.append( template.replace("$ARG1",w1).replace("$ARG2",w4) + ' ' + w1 )
    sent_arr.append( template.replace("$ARG1",w1).replace("$ARG2",w4) + ' ' + w4 )
    sent_arr.append( template.replace("$ARG2",w1).replace("$ARG1",w4) + ' ' + w1 )
    sent_arr.append( template.replace("$ARG2",w1).replace("$ARG1",w4) + ' ' + w4 )
    return sent_arr

def insert_into_template(input_data):
    template_arr = template_nouns
    input_sent = []
    input_template_id = []
    for i in range(len(input_data)):
        w1, w2 = input_data[i]
        for j, template in enumerate(template_arr):
            w1_w2_list = create_data(template, w1, w2)
            input_sent += w1_w2_list
            input_template_id += [j]* len(w1_w2_list) 
    return input_sent, input_template_id

def store_text_output(store_data, output_file_name):
    with open(output_file_name, 'w') as f_out:
        for sent in store_data:
            f_out.write(sent+'\n')

def rfind_list(arr, search_item):
    return len(arr) - 1 - arr[::-1].index(search_item)
    
def create_tensor_output(input_sent, input_template_id, tokenizer_GPT2):
    #with open(output_file_name, 'w') as f_out:
    sent_len_list = []
    target_position_list = []
    sent_idx_list = []
    template_last_arr = tempaate_nouns_last_arr
    for template_id, sent in zip(input_template_id, input_sent):
        sent_idx = tokenizer_GPT2.encode(sent, add_prefix_space=True)
        sent_idx_list.append(sent_idx)
        sent_len_list.append(len(sent_idx))
        last_word_in_prompt_idx = template_last_arr[template_id]
        assert last_word_in_prompt_idx in sent_idx, print(last_word_in_prompt_idx, sent_idx, sent)
        end_of_prompt_idx = rfind_list(sent_idx, last_word_in_prompt_idx)
        assert end_of_prompt_idx+1 < len(sent_idx)
        target_position_list.append(end_of_prompt_idx+1)
    max_length = max(sent_len_list)
    sent_tensor = torch.zeros( (len(sent_len_list), max_length), dtype = torch.int32)
    sent_len_tensor = torch.tensor(sent_len_list, dtype = torch.int32)
    target_position_tensor = torch.tensor(target_position_list, dtype = torch.int32)
    for i in range(len(sent_len_list)):
        sent_tensor[i,:sent_len_list[i]] = torch.tensor(sent_idx_list[i], dtype = torch.int32)
    return sent_tensor, sent_len_tensor, target_position_tensor


for f_name in os.listdir(input_folder):
    print(f_name)
    fields = f_name.split('_')
    split_name = '_'.join(fields[2:])
    input_data = load_input( os.path.join(input_folder, f_name) )
    input_sent, input_template_id = insert_into_template(input_data)
    store_text_output(input_sent, os.path.join(output_text_dir, f_name))
    sent_tensor, sent_len_tensor, target_position_tensor = create_tensor_output(input_sent, input_template_id, tokenizer_GPT2) 
    output_file_name = os.path.join(output_tensor_dir, f_name)
    torch.save([sent_tensor, sent_len_tensor, target_position_tensor], output_file_name)
        
