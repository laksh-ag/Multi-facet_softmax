import os
import sys
sys.path.insert(0, sys.path[0]+'/../..')

import torch
from gpt2_model.tokenization_gpt2 import GPT2Tokenizer

template_people = ['The $ARG1 and the $ARG2 are in front of me. I go forward and talk to the',
                   "The $ARG1 and the $ARG2 are my favorite, and I especially love the",
                   "The $ARG1 and the $ARG2 happily live together. One day, bad luck happens to the",
                   "The $ARG1 and the $ARG2 stay at their house, and one of them feels hungry, who is the"]
template_location = ["I went to $ARG1 and $ARG2 before, and I love one of the places more, which is",
                     "$ARG1 and $ARG2 are my favorite, and I especially love",
                     "My uncle used to live in $ARG1 and $ARG2 and he sold his house in",
                     "The traveler plans to visit $ARG1 and $ARG2, and the traveler first arrives in"]

stop_word_set = set(['he', 'she', 'her', 'his'])

input_folder = "data/processed/analogy_google/split"

w1_w3_baseline = True
#w1_w3_baseline = False

if w1_w3_baseline:
    output_text_dir = "data/processed/analogy_google/text_more_w1w3"
    output_tensor_dir = "data/processed/analogy_google/tensor_more_w1w3"
else:
    output_text_dir = "data/processed/analogy_google/text_more"
    output_tensor_dir = "data/processed/analogy_google/tensor_more"


#output_text_dir = "data/processed/analogy_google/text_w1w3"
#output_tensor_dir = "data/processed/analogy_google/tensor_w1w3"
#output_text_dir = "data/processed/analogy_google/text"
#output_tensor_dir = "data/processed/analogy_google/tensor"

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

template_people_last_arr = get_last_id(template_people)
template_location_last_arr = get_last_id(template_location)

def load_input(input_path):
    input_data = []
    with open(input_path) as f_in:
        for line in f_in:
            w1, w2, w3, w4 = line.rstrip().split()
            input_data.append([w1, w2, w3, w4])
    return input_data

def create_data(template, w1, w4):
    sent_arr = []  
    if w1 in stop_word_set or w4 in stop_word_set:
        return sent_arr
    assert template.split()[-1] != w1
    assert template.split()[-1] != w4
    sent_arr.append( template.replace("$ARG1",w1).replace("$ARG2",w4) + ' ' + w1 )
    sent_arr.append( template.replace("$ARG1",w1).replace("$ARG2",w4) + ' ' + w4 )
    sent_arr.append( template.replace("$ARG2",w1).replace("$ARG1",w4) + ' ' + w1 )
    sent_arr.append( template.replace("$ARG2",w1).replace("$ARG1",w4) + ' ' + w4 )
    return sent_arr

def insert_into_template(input_data, word_type):
    if word_type == 'location':
        template_arr = template_location
    elif word_type == 'people':
        template_arr = template_people
    input_sent = []
    input_template_id = []
    for i in range(len(input_data)):
        w1, w2, w3, w4 = input_data[i]
        for j, template in enumerate(template_arr):
            if not w1_w3_baseline:
                w1_w4_list = create_data(template, w1, w4)
                w2_w3_list = create_data(template, w2, w3)
                input_sent += w1_w4_list + w2_w3_list
                input_template_id += [j]* ( len(w1_w4_list) + len(w2_w3_list) )
            else:
                w1_w2_list = create_data(template, w1, w2)
                w3_w4_list = create_data(template, w3, w4)
                w1_w3_list = create_data(template, w1, w3)
                w2_w4_list = create_data(template, w2, w4)
                input_sent += w1_w2_list + w3_w4_list + w1_w3_list + w2_w4_list
                input_template_id += [j]* ( len(w1_w2_list) + len(w3_w4_list) +len(w1_w3_list) + len(w2_w4_list)  )
            #w1_w4_list = create_data(template, w1, w3)
            #w2_w3_list = create_data(template, w2, w4)
    return input_sent, input_template_id

def store_text_output(store_data, output_file_name):
    with open(output_file_name, 'w') as f_out:
        for sent in store_data:
            f_out.write(sent+'\n')

def rfind_list(arr, search_item):
    return len(arr) - 1 - arr[::-1].index(search_item)
    
def create_tensor_output(input_sent, input_template_id, tokenizer_GPT2, word_type):
    #with open(output_file_name, 'w') as f_out:
    sent_len_list = []
    target_position_list = []
    sent_idx_list = []
    if word_type == 'location':
        template_last_arr = template_location_last_arr
    elif word_type == 'people':
        template_last_arr = template_people_last_arr
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


for split_ratio in os.listdir(input_folder):
    print(split_ratio)
    output_text_split_dir = os.path.join(output_text_dir, split_ratio)
    output_tensor_split_dir = os.path.join(output_tensor_dir, split_ratio)
    if not os.path.exists(output_text_split_dir):
        os.makedirs(output_text_split_dir)
    if not os.path.exists(output_tensor_split_dir):
        os.makedirs(output_tensor_split_dir)
    for f_name in os.listdir(os.path.join(input_folder, split_ratio)):
        print(f_name)
        fields = f_name.split('_')
        word_type = fields[0]
        relation_name = fields[1]
        split_name = '_'.join(fields[2:])
        input_data = load_input( os.path.join(input_folder, split_ratio, f_name) )
        input_sent, input_template_id = insert_into_template(input_data, word_type)
        store_text_output(input_sent, os.path.join(output_text_dir, split_ratio, f_name))
        sent_tensor, sent_len_tensor, target_position_tensor = create_tensor_output(input_sent, input_template_id, tokenizer_GPT2, word_type) 
        output_file_name = os.path.join(output_tensor_dir, split_ratio, f_name)
        torch.save([sent_tensor, sent_len_tensor, target_position_tensor], output_file_name)
            
