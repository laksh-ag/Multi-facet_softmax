import os, sys
import numpy as np
import torch
import torch.nn as nn
import time
import math

from utils import seed_all_randomness, load_corpus, str2bool
import utils_testing

from gpt2_model.tokenization_gpt2 import GPT2Tokenizer
from gpt2_model.modeling_gpt2_multi import GPT2MultiLMHeadModel, GPT2MoSLMHeadModel, GPT2LMHeadModel, GPT2Model
from gpt2_model.configuration_gpt2 import GPT2Config
import argparse

parser = argparse.ArgumentParser(description='PyTorch Train Future Topic Prediction')


def add_model_options(parser, suffix):
    parser.add_argument('--model_path'+suffix, type=str,  default='./models/',
                        help='path to load the model')
    parser.add_argument('--load_file_name'+suffix, type=str,  default='LM_weights.pt',
                    help='file name of saved model')
    parser.add_argument('--n_facet'+suffix, type=int, default=5,
                        help='number of facets')
    parser.add_argument('--n_facet_hidden'+suffix, type=int, default=0,
                        help='number of facets')
    parser.add_argument('--n_facet_MLP'+suffix, type=int, default=0,
                        help='size of compression layer')
    parser.add_argument('--n_facet_window'+suffix, type=int, default=0,
                        help='size of windows we look at')
    parser.add_argument('--n_facet_effective'+suffix, type=int, default=1,
                        help='number of facet heads')

    parser.add_argument('--use_avg'+suffix, type=str2bool, nargs='?', default=False,
                        help='Whether we want to add an average embedding term to stablize the training')
    parser.add_argument('--use_MoS'+suffix, type=str2bool, nargs='?', default=True,
                        help='Whether we want to do the normalization for each facet (i.e., use mixture of softmax)')
    parser.add_argument('--weight_mode'+suffix, type=str,  default='dynamic',
                        help='could be empty, dynamic, and statis')
    parser.add_argument('--use_proj_bias'+suffix, type=str2bool, nargs='?', default=True,
                        help='Whether we want to add an bias term in the linear projection layer')
    parser.add_argument('--efficient_mode'+suffix, type=str,  default='None',
                        help='how to save computational time')
    parser.add_argument('--masking_ratio'+suffix, type=float, default=-1,
                        help='dynamically use single facets. Use -1 to turn off this efficient mode')
    parser.add_argument('--last_num'+suffix, type=int, default=0,
                        help='number of facet that does not have multiple partitions')
    

parser.add_argument('--data', type=str, default='./data/processed/wiki2016_gpt2/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_all_min100',
                    help='location of the data corpus')
parser.add_argument('--outf', type=str, default='gen_log/generated.txt',
                    help='output file for generated text')

parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=256,
                    help='sequence length')
parser.add_argument('--max_batch_num', type=int, default=100, 
                    help='number of batches for evaluation')
parser.add_argument('--num_sent_gen', type=int, default=3, metavar='N',
                    help='In each prompt, generate how many sentences')
parser.add_argument('--gen_sent_len', type=int, default=50, metavar='N',
                    help='In each prompt, generate sentences with length gen_sent_len')

parser.add_argument('--run_eval', type=str2bool, nargs='?', default=True,
                    help='If false, we only print the results')

parser.add_argument('--cuda', type=str2bool, nargs='?', default=True,
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

add_model_options(parser, "_multi")
add_model_options(parser, "_single")

args = parser.parse_args()

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed,args.cuda)

print('Args: {}'.format(args))

device = torch.device("cuda" if args.cuda else "cpu")

skip_training = True
dataloader_train_arr, dataloader_val = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, device, args.tensor_folder, split_num = 1, skip_training = skip_training)

#model_name = 'gpt2-medium'
model_name = 'gpt2'
gpt2_config = GPT2Config.from_pretrained(model_name)
gpt2_config.output_hidden_states = True



def load_model(model_path, gpt2_config, n_facet, n_facet_window, n_facet_hidden, n_facet_MLP, n_facet_effective, weight_mode, use_avg, use_MoS, use_proj_bias, efficient_mode, last_num, device):
    LM_state_dict = torch.load(os.path.join(model_path, 'LM_weights.pt'), map_location=device)
    #GPT2_LM = GPT2MultiLMHeadModel.from_pretrained(model_name, state_dict = LM_state_dict)
    GPT2_encoder = GPT2Model(gpt2_config)
    if use_MoS:
        #weight_mode = 'dynamic'
        #weight_mode = 'static'
        #weight_mode = ''
        #GPT2_LM = GPT2MoSLMHeadModel(gpt2_config, GPT2_encoder, n_facet, n_facet_hidden, weight_mode, use_proj_bias)
        GPT2_LM = GPT2MoSLMHeadModel(gpt2_config, GPT2_encoder, n_facet, n_facet_hidden, weight_mode, use_proj_bias,n_facet_window = n_facet_window, n_facet_MLP = n_facet_MLP, efficient_mode=efficient_mode, device=device, n_facet_effective_in=n_facet_effective, last_num=last_num)
    else:
        GPT2_LM = GPT2MultiLMHeadModel(gpt2_config, GPT2_encoder, n_facet, n_facet_hidden, use_avg)
    GPT2_LM.load_state_dict(LM_state_dict)
    if args.cuda:
        GPT2_LM = GPT2_LM.cuda()
    return GPT2_LM


model_multi = load_model(args.model_path_multi, gpt2_config, args.n_facet_multi, args.n_facet_window_multi, args.n_facet_hidden_multi, args.n_facet_MLP_multi, args.n_facet_effective_multi, args.weight_mode_multi, args.use_avg_multi, args.use_MoS_multi, args.use_proj_bias_multi, args.efficient_mode_multi, args.last_num_multi, device)

model_single = load_model(args.model_path_single, gpt2_config, args.n_facet_single, args.n_facet_window_single, args.n_facet_hidden_single, args.n_facet_MLP_single, args.n_facet_effective_single, args.weight_mode_single, args.use_avg_single, args.use_MoS_single, args.use_proj_bias_single, args.efficient_mode_single, args.last_num_single, device)

device_gpt2 = "cuda:0" if torch.cuda.is_available() else "cpu"
gpt2_org_model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device_gpt2)

tokenizer_GPT2 = GPT2Tokenizer.from_pretrained('gpt2')

with open(args.outf, 'w') as outf:
    utils_testing.visualize_interactive_LM(model_multi, model_single, gpt2_org_model, device, args.num_sent_gen, args.gen_sent_len, dataloader_val, outf, args.max_batch_num, tokenizer_GPT2, args.bptt, readable_context = False, run_eval = args.run_eval)
