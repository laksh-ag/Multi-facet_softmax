import os
import sys
sys.path.insert(0, sys.path[0]+'/../..')
from gpt2_model.modeling_gpt2_multi import GPT2MoSLMHeadModel, GPT2Model
from gpt2_model.configuration_gpt2 import GPT2Config
from gpt2_model.tokenization_gpt2 import GPT2Tokenizer
from scipy import spatial
import torch

input_path = "./resources/hypernym_benchmarks/related_word_list"
output_path = "./resources/hypernym_benchmarks/word_sim_list"

model_name= 'gpt2'
gpt2_config = GPT2Config.from_pretrained(model_name)

#save = "./models/gpt2_wiki_n1_init-20210316-011244"
#load_file_name = "LM_weights_8.pt"
save = "./models/gpt2_wiki2021_n3_init-20210417-013247"
load_file_name = "LM_weights_1_7.pt"

GPT2_encoder = GPT2Model(gpt2_config)
weight_mode = 'dynamic'
use_proj_bias = True
n_facet = 1
n_facet_hidden = 1
n_facet_window = 0
n_facet_MLP = 0
#n_facet_hidden = 3
#n_facet_window = -2
#n_facet_MLP = -1

tokenizer_GPT2 = GPT2Tokenizer.from_pretrained('gpt2')

GPT2_LM = GPT2MoSLMHeadModel(gpt2_config, GPT2_encoder, n_facet, n_facet_hidden, weight_mode, use_proj_bias, n_facet_window, n_facet_MLP)

LM_state_dict = torch.load(os.path.join( save, load_file_name), map_location='cpu')
GPT2_LM.load_state_dict(LM_state_dict)
del LM_state_dict

def load_input(input_path, tokenizer_GPT2):
    input_data = []
    with open(input_path) as f_in:
        for line in f_in:
            w1, w2, hyper, rel = line.rstrip().split()
            #w1_w, w1_pos = w1.split('-')
            #w2_w, w2_pos = w2.split('-')
            w1_w, w1_pos = w1[:-2], w1[-1]
            w2_w, w2_pos = w2[:-2], w2[-1]
            if w1_pos != 'n' or w2_pos != 'n' or w1_w == w2_w:
                continue
            w1_idx = tokenizer_GPT2.encode(w1_w, add_prefix_space=True)
            #if len(w1_idx) > 1:
            #    print("skipping {}".format(w1_w))
            #    continue
            w2_idx = tokenizer_GPT2.encode(w2_w, add_prefix_space=True)
            #if len(w2_idx) > 1:
            #    print("skipping {}".format(w2_w))
            #    continue
            if w1_idx[0] == w2_idx[0]:
                continue
            input_data.append([w1_w, w2_w, w1_idx[0], w2_idx[0]])
    return input_data

input_data = load_input(input_path, tokenizer_GPT2)

output_data = []
output_word_embedding = GPT2_LM.lm_head.weight.data
for w1, w2, w1_idx, w2_idx in input_data:
    w1_emb = output_word_embedding[w1_idx,:].tolist()
    w2_emb = output_word_embedding[w2_idx,:].tolist()
    cosine_sim = 1 - spatial.distance.cosine(w1_emb, w2_emb)
    output_data.append([w1,w2,str(cosine_sim)])

output_data_sorted = sorted(output_data, key = lambda x: x[2], reverse=True)

with open(output_path, 'w') as f_out:
    f_out.write( '\n'.join([' '.join(x) for x in output_data_sorted] ) )
#sort based on similarity
