import os, sys
import numpy as np
#import scipy.stats
#from scipy import linalg
from evaluation import comprehensive_evaluate, simple_eval, only_test_time_loss
#import gc

import torch
import torch.nn as nn
import time
import math
import io

from utils import seed_all_randomness, create_exp_dir, load_corpus, str2bool, load_sent_corpus

from gpt2_model.tokenization_gpt2 import GPT2Tokenizer
from gpt2_model.modeling_gpt2_multi import GPT2MultiLMHeadModel, GPT2MoSLMHeadModel, GPT2LMHeadModel, GPT2Model
from gpt2_model.configuration_gpt2 import GPT2Config
from gpt2_model.optimization import AdamW
import argparse

parser = argparse.ArgumentParser(description='PyTorch Train Future Topic Prediction')
parser.add_argument('--data', type=str, default='./data/processed/wiki2016_gpt2/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_all_min100',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,  default='./models/',
                    help='path to save the final model')
parser.add_argument('--load_file_name', type=str,  default='LM_weights.pt',
                    help='file name of saved model')

parser.add_argument('--model_name', type=str,  default='gpt2',
                    help='pretrained model name')

parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=256,
                    help='sequence length')
parser.add_argument('--n_facet', type=int, default=5,
                    help='number of facets')
parser.add_argument('--n_facet_hidden', type=int, default=1,
                    help='number of facets')
parser.add_argument('--n_facet_window', type=int, default=0,
                    help='number of facets')
parser.add_argument('--n_facet_MLP', type=int, default=0,
                    help='number of facets')
parser.add_argument('--efficient_mode', type=str,  default='None',
                    help='how to save computational time')
parser.add_argument('--softmax_nonlinear', type=str,  default='None',
                    help='Whether adding some nonlinearity into the softmax layer')
parser.add_argument('--masking_ratio', type=float, default=-1,
                    help='dynamically use single facets. Use -1 to turn off this efficient mode')
parser.add_argument('--last_num', type=int, default=0,
                    help='number of facet that does not have multiple partitions')


parser.add_argument('--random_init', type=str2bool, nargs='?', default=True,
                    help='initialize the model randomly')
parser.add_argument('--use_avg', type=str2bool, nargs='?', default=False,
                    help='Whether we want to add an average embedding term to stablize the training')
parser.add_argument('--use_proj_bias', type=str2bool, nargs='?', default=True,
                    help='Whether we want to add an bias term in the linear projection layer')
parser.add_argument('--use_MoS', type=str2bool, nargs='?', default=False,
                    help='Whether we want to do the normalization for each facet (i.e., use mixture of softmax)')
parser.add_argument('--individual_hidden_states', type=str2bool, nargs='?', default=False,
                    help='Whether we want to use one layer for one facet')

parser.add_argument('--template_experiment', type=str2bool, nargs='?', default=False,
                    help='only testing the learnability of some templates')
parser.add_argument('--relation_type', type=str,  default='location_capital-common-countries',
                    help='template file prefix that indicates the relation between the words')
parser.add_argument('--testing_file_names', type=str,  default='',
                    help='template file prefix that is used to test the model. If empty, using relation_type. If having multiple files, use | to separate the names')
parser.add_argument('--only_visualize', type=str2bool, nargs='?', default=False,
                    help='print out the predictions')
parser.add_argument('--only_eval', type=str2bool, nargs='?', default=False,
                    help='only do more comprehensive evaluation')
parser.add_argument('--simple_eval', type=str2bool, nargs='?', default=False,
                    help='only do simple evaluation')
parser.add_argument('--time_loss_eval', type=str2bool, nargs='?', default=False,
                    help='only test time and loss')
parser.add_argument('--eval_batch_num', type=int, default=1000,
                    help='If --only_eval True, we want to evaluate our model using how many batches')
parser.add_argument('--reconstruction_file_name', type=str, default="",
                    help='If --only_eval True and --eval_recon_top_k > 1, we will store the reconstruction error and loss into this file')
parser.add_argument('--eval_recon_top_k', type=int, default=0,
                    help='When --only_eval True and --eval_recon_top_k > 1, we want to compute the correlation between the reconstruction error and loss improvement')
parser.add_argument('--template_vis_file_name', type=str, default="",
                    help='When the template_experiment is True, we will store the predictions every save_every_n_valid epochs')

parser.add_argument('--optimizer', type=str, default="AdamW",
                    help='optimization algorithm. Could be SGD or Adam')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')

parser.add_argument('--valid_per_epoch', type=int, default=2,
                    help='Number of times we want to run through validation data and save model within an epoch')
parser.add_argument('--training_split_num', type=int, default=1,
                    help='We want to split training corpus into how many subsets. Splitting training dataset seems to make pytorch run much faster and we can store and eval the model more frequently')

parser.add_argument('--save_every_n_valid', type=int, default=8,
                    help='save another model after every n validation')
parser.add_argument('--start_training_split', type=int, default=0,
                    help='We want to split training corpus into how many subsets. Splitting training dataset seems to make pytorch run much faster and we can store and eval the model more frequently')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, nargs='?', default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')

args = parser.parse_args()


if not args.continue_train:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

if not args.only_visualize and not args.only_eval and not args.simple_eval and not args.time_loss_eval and not args.template_experiment:
    create_exp_dir(args.save, scripts_to_save=['./src/main.py', './src/gpt2_model/modeling_gpt2_multi.py'])

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
        sys.stdout.flush()
    if log_:
        file_name = os.path.join(args.save, 'log.txt')
        if args.template_experiment and len(args.reconstruction_file_name):
            file_name = args.reconstruction_file_name + '_log'
        with open(file_name, 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed,args.cuda)

num_gpu = torch.cuda.device_count()

logging('Args: {}'.format(args))
logging('GPU number: {}'.format(num_gpu))

device = torch.device("cuda" if args.cuda else "cpu")
if args.efficient_mode == 'even_multi_shuffle':
    n_facet_effective = 3
elif args.efficient_mode == 'even_multi_alter':
    n_facet_effective = 3
elif args.efficient_mode == 'even_last_2':
    n_facet_effective = 3
elif args.efficient_mode == 'even':
    n_facet_effective = 1
elif args.efficient_mode == 'even_last':
    n_facet_effective = 2
elif args.efficient_mode == 'context_merge':
    n_facet_effective = args.n_facet - 1
else:
    n_facet_effective = args.n_facet

skip_training = False
if args.only_visualize or args.only_eval or args.simple_eval or args.time_loss_eval:
    skip_training = True

dataloader_test_arr = []
testing_file_names_list = args.testing_file_names.split('+')
if args.template_experiment:
    dataloader_train_arr, dataloader_val, dataloader_test_arr = load_sent_corpus(args.data, args.relation_type, testing_file_names_list, args.batch_size, args.batch_size, device, skip_training = skip_training)
elif args.time_loss_eval or args.simple_eval:
    dataloader_train_arr, dataloader_val, dataloader_test = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, device, args.tensor_folder, args.training_split_num, skip_training = skip_training, load_testing=True)
else:
    dataloader_train_arr, dataloader_val = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, device, args.tensor_folder, args.training_split_num, skip_training = skip_training)


#model_name = 'gpt2-medium'
#model_name = 'gpt2'
model_name= args.model_name
gpt2_config = GPT2Config.from_pretrained(model_name)
gpt2_config.output_hidden_states = True
if not args.continue_train:
    if args.random_init:
        if args.n_facet == 0:
            GPT2_LM = GPT2LMHeadModel(gpt2_config)
        else:
            GPT2_encoder = GPT2Model(gpt2_config)
            if args.use_MoS:
                weight_mode = 'dynamic'
                #weight_mode = 'static'
                #weight_mode = ''
                GPT2_LM = GPT2MoSLMHeadModel(gpt2_config, GPT2_encoder, args.n_facet, args.n_facet_hidden, weight_mode, args.use_proj_bias)
            else:
                GPT2_LM = GPT2MultiLMHeadModel(gpt2_config, GPT2_encoder, args.n_facet, args.n_facet_hidden, args.use_avg)
            
    else:
        if args.n_facet == 0:
            GPT2_LM = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            GPT2_LM_init = GPT2LMHeadModel.from_pretrained(model_name)

            GPT2_encoder = GPT2Model.from_pretrained(model_name, config = gpt2_config)
            if args.use_MoS:
                weight_mode = 'dynamic'
                #weight_mode = 'static'
                #weight_mode = ''
                GPT2_LM = GPT2MoSLMHeadModel(gpt2_config, GPT2_encoder, args.n_facet, args.n_facet_hidden, weight_mode, args.use_proj_bias, args.n_facet_window, args.n_facet_MLP, efficient_mode=args.efficient_mode, masking_ratio = args.masking_ratio, seq_len = args.bptt, device=device, n_facet_effective_in=n_facet_effective, last_num=args.last_num, softmax_nonlinear=args.softmax_nonlinear, individual_hidden_states = args.individual_hidden_states)
            else:
                GPT2_LM = GPT2MultiLMHeadModel(gpt2_config, GPT2_encoder, args.n_facet, args.n_facet_hidden, args.use_avg)
            
            softmax_layer = GPT2_LM.get_output_embeddings()
            softmax_layer.weight.data = GPT2_LM_init.get_output_embeddings().weight.data
            del GPT2_LM_init
else:
    #LM_state_dict = torch.load(os.path.join(args.save, args.load_file_name), map_location=device)
    LM_state_dict = torch.load(os.path.join(args.save, args.load_file_name), map_location='cpu')
    #GPT2_LM = GPT2MultiLMHeadModel.from_pretrained(model_name, state_dict = LM_state_dict)
    GPT2_encoder = GPT2Model(gpt2_config)
    if args.use_MoS:
        weight_mode = 'dynamic'
        #weight_mode = 'static'
        #weight_mode = ''
        vis_seq_loss = False
        vis_simple = False
        only_commpute_loss = False
        if args.simple_eval:
            vis_simple = True
        elif args.only_eval:
            vis_seq_loss = True
        elif args.time_loss_eval:
            only_commpute_loss = True
        GPT2_LM = GPT2MoSLMHeadModel(gpt2_config, GPT2_encoder, args.n_facet, args.n_facet_hidden, weight_mode, args.use_proj_bias, args.n_facet_window, args.n_facet_MLP, vis_seq_loss, vis_simple = vis_simple, only_commpute_loss=only_commpute_loss, efficient_mode=args.efficient_mode, masking_ratio = args.masking_ratio, seq_len = args.bptt, device=device, n_facet_effective_in=n_facet_effective, last_num=args.last_num, softmax_nonlinear=args.softmax_nonlinear, individual_hidden_states = args.individual_hidden_states)
    else:
        GPT2_LM = GPT2MultiLMHeadModel(gpt2_config, GPT2_encoder, args.n_facet, args.n_facet_hidden, args.use_avg)
    GPT2_LM.load_state_dict(LM_state_dict)
    del LM_state_dict

#torch.cuda.empty_cache()
#gc.collect()

if args.template_experiment:
    GPT2_LM.lm_head.weight.requires_grad = False

if args.cuda:
    if args.single_gpu:
        parallel_GPT2_LM = GPT2_LM.cuda()
    else:
        #parallel_GPT2_LM = nn.parallel.DistributedDataParallel(GPT2_LM, dim=0).cuda()
        parallel_GPT2_LM = nn.DataParallel(GPT2_LM, dim=0).cuda()
else:
    parallel_GPT2_LM = GPT2_LM


def visualize_output(outputs, weight, tokenizer_GPT2, f_out, feature, visualize_idx_list, dynamic_length = False):
    batch_size = feature.size(0)
    stacked_facet_lm_logits = outputs[2]
    pred_prob = outputs[1]
    #(n_facet, bsz, seq_len, vocab_size)
    n_facet = stacked_facet_lm_logits.size(0)

    feature_text = [ [tokenizer_GPT2._convert_id_to_token(x) for x in feature[i,:].tolist()] for i in range(feature.size(0))]
    for i_sent in range(batch_size):
        if dynamic_length:
            word_idx_arr = [visualize_idx_list[i_sent]]
        else:
            word_idx_arr = visualize_idx_list
        for word_idx in word_idx_arr:
            f_out.write(tokenizer_GPT2.convert_tokens_to_string(feature_text[i_sent][:(word_idx+2)])+'\n')
            #f_out.write("weight:" + ",".join(weight[i_sent,word_idx,:].tolist())+'\n')
            topj = 5
            #print(pred_prob.size())
            top_value, top_idx = torch.topk( pred_prob[i_sent, word_idx,:], k=topj, dim=0 )
            f_out.write(' '.join([tokenizer_GPT2._convert_id_to_token(top_idx[j].item())+'|'+str(top_value[j].item()) for j in range(topj)]))
            f_out.write('\n')

            top_value, top_idx = torch.topk( stacked_facet_lm_logits[:, i_sent, word_idx,:], k=topj, dim=1 )
            for k in range(n_facet):
                f_out.write('k'+str(k)+': '+str(weight[i_sent,word_idx,k].item())+'|')
                #f_out.write('k'+str(k)+': ')
                #print(top_idx[k,:])
                #print(top_value[k,:])
                f_out.write(' '.join([tokenizer_GPT2._convert_id_to_token(top_idx[k,j].item())+'|'+str(top_value[k,j].item()) for j in range(topj)]))
                f_out.write('\n')
            f_out.write('\n')
        f_out.write('\n')

def evaluate(dataloader, str_out):
    # Turn on evaluation mode which disables dropout.
    GPT2_LM.eval()
    total_loss = 0.
    total_div = 0.
    dataset_size = 0
    #total_count_arr = np.zeros(n_facet_effective)
    total_weights_arr = np.zeros(n_facet_effective)
    total_count_best_arr = np.zeros(n_facet_effective)
    

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if args.template_experiment:
                feature, sent_len, target_position = sample_batched
                labels = -100 * torch.ones_like(feature)
                #labels[target_position:sent_len] = feature[target_position:sent_len]
                for j in range(feature.size(0)): #should be able to use gather and scatter if batch size is large
                    labels[j,target_position[j]:sent_len[j]] = feature[j,target_position[j]:sent_len[j]]
                    #labels[j,target_position[j]] = feature[j,target_position[j]]
            else:
                feature,labels = sample_batched
            batch_size = feature.size(0)
            if args.n_facet == 0:
                outputs = parallel_GPT2_LM(feature, labels=feature)
            else:
                #outputs, emb_div, count_arr, count_best_arr, weight = parallel_GPT2_LM(feature, labels=feature)
                # print(feature.max())
                # print(feature.min())
                # print(labels.max())
                # print(labels.min())
                outputs, emb_div, count_best_arr, weight = parallel_GPT2_LM(feature, labels=labels)
                if str_out is not None and i_batch % 10 == 0:
                    visualize_output(outputs, weight, tokenizer_GPT2, str_out, feature, [x-1 for x in target_position.cpu().detach().tolist()], dynamic_length=True)

                #if num_gpu > 1:
                emb_div = emb_div.mean()
                #count_arr = count_arr.mean(dim=0)
                weights_arr = weight.mean(dim=1).mean(dim=0)
                count_best_arr = count_best_arr.mean(dim=0)
                total_div += emb_div.item() * batch_size
                #total_count_arr += count_arr.cpu().numpy() * batch_size
                total_weights_arr += weights_arr.cpu().detach().numpy() * batch_size
                total_count_best_arr += count_best_arr.cpu().numpy() * batch_size
            loss = outputs[0]
            if num_gpu > 1:
                loss = loss.mean()
            total_loss += loss.item() * batch_size
            dataset_size += batch_size

    #data_size = len(dataloader.dataset)
    #return total_loss / data_size, total_div / data_size, total_count_arr / data_size, total_count_best_arr / data_size
    return total_loss / dataset_size, total_div / dataset_size, total_count_best_arr / dataset_size, total_weights_arr / dataset_size, str_out

def train_one_epoch(dataloader_train, split_i):
    start_time = time.time()
    total_loss = 0.
    total_loss_epoch = 0.
    total_batch_size = 0
    total_div = 0.
    #total_count_arr = np.zeros(n_facet_effective)
    total_weights_arr = np.zeros(n_facet_effective)
    total_count_best_arr = np.zeros(n_facet_effective)

    GPT2_LM.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
        if args.template_experiment:
            feature, sent_len, target_position = sample_batched
            labels = -100 * torch.ones_like(feature)
            for j in range(feature.size(0)): #should be able to use gather and scatter if batch size is large
                labels[j,target_position[j]:sent_len[j]] = feature[j,target_position[j]:sent_len[j]]
                #labels[j,target_position[j]] = feature[j,target_position[j]]
        else:
            feature,labels = sample_batched
        if args.n_facet == 0:
            outputs = parallel_GPT2_LM(feature, labels=feature)
        else:
            outputs, emb_div, count_best_arr, weight = parallel_GPT2_LM(feature, labels=labels, output_weight=True)
            #if num_gpu > 1:
            emb_div = emb_div.mean()
            #count_arr = count_arr.mean(dim=0)
            weights_arr = weight.mean(dim=1).mean(dim=0)
            count_best_arr = count_best_arr.mean(dim=0)
            total_div += emb_div.item()
            #total_count_arr += count_arr.cpu().numpy()
            total_weights_arr += weights_arr.cpu().detach().numpy() 
            total_count_best_arr += count_best_arr.cpu().numpy()
        loss = outputs[0]
        if num_gpu > 1:
            loss = loss.mean()
        
        total_loss += loss.item()
        batch_size = feature.size(0)
        total_loss_epoch += loss.item() * batch_size 
        total_batch_size += batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(GPT2_LM.parameters(), args.clip)
        optimizer.step()
        if (i_batch % args.log_interval == args.log_interval - 1):
            cur_loss = total_loss / args.log_interval
            cur_div = total_div / args.log_interval
            #cur_count_arr = total_count_arr / args.log_interval
            cur_weights_arr = total_weights_arr / args.log_interval
            cur_count_best_arr = total_count_best_arr / args.log_interval
            elapsed = time.time() - start_time
            logging('| e {:3d} {:3d} | {:5d}/{:5d} b | ms/batch {:5.2f} | div {:5.3f} | weights {} | prob_best_arr {} |'
                    'l {:5.5f}  '.format(
                epoch, split_i, i_batch, len(dataloader_train.dataset) // args.batch_size, 
                #elapsed * 1000 / args.log_interval, cur_div, cur_count_arr/np.sum(cur_count_arr), cur_count_best_arr/np.sum(cur_count_best_arr), cur_loss))
                elapsed * 1000 / args.log_interval, cur_div, cur_weights_arr , cur_count_best_arr/np.sum(cur_count_best_arr), cur_loss))

            total_loss = 0.
            total_div = 0.
            #total_count_arr = np.zeros(n_facet_effective)
            total_weights_arr = np.zeros(n_facet_effective)
            start_time = time.time()
    if i_batch % args.log_interval != 0:
        cur_loss = total_loss / (i_batch % args.log_interval+1)
        cur_div = total_div / (i_batch % args.log_interval+1)
        cur_weights_arr = total_weights_arr / (i_batch % args.log_interval+1)
        cur_count_best_arr = total_count_best_arr / (i_batch % args.log_interval+1)
        elapsed = time.time() - start_time
        logging('| e {:3d} {:3d} | {:5d}/{:5d} b | ms/batch {:5.2f} | div {:5.3f} | weights {} | prob_best_arr {} |'
                'l {:5.5f}  '.format(
            epoch, split_i, i_batch, len(dataloader_train.dataset) // args.batch_size, 
            elapsed * 1000 / args.log_interval, cur_div, cur_weights_arr , cur_count_best_arr/np.sum(cur_count_best_arr), cur_loss))
    return total_loss_epoch / float(total_batch_size)



def only_visualize(dataloader, f_out):
    # Turn on evaluation mode which disables dropout.
    GPT2_LM.eval()
    max_batch_num = 10
    tokenizer_GPT2 = GPT2Tokenizer.from_pretrained(model_name)
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if args.template_experiment:
                feature, sent_len, target_position = sample_batched
                labels = -100 * torch.ones_like(feature)
                labels[target_position:sent_len] = feature[target_position:sent_len]
            else:
                feature = sample_batched
                labels = feature
            seq_len = feature.size(1)
            #outputs, emb_div, count_arr, count_best_arr, weight = parallel_GPT2_LM(feature, labels=feature)
            outputs, emb_div, count_best_arr, weight = parallel_GPT2_LM(feature, labels=labels)
            print(i_batch)
            sys.stdout.flush()
            visualize_output(outputs, weight, tokenizer_GPT2, f_out, feature, list(range(10, seq_len - 2)))

            if i_batch >= max_batch_num:
                break
#(loss), lm_logits, stacked_facet_lm_logits , presents, (all hidden_states), (attentions)

    #data_size = len(dataloader.dataset)

def print_parameters(model, excluding_prefix_arr = None):
    parameter_sum = 0
    for name, param in model.named_parameters():
        contiue_or_not = False
        for excluding_prefix in excluding_prefix_arr:
            if excluding_prefix is not None and excluding_prefix == name[:len(excluding_prefix)]:
                print("Skipping ", name, param.numel())
                contiue_or_not = True
                break
        if contiue_or_not:
            continue
        if param.requires_grad:
            print(name, param.numel())
            parameter_sum += param.numel()
    return parameter_sum

if args.time_loss_eval:
    assert not args.template_experiment
    total_params = sum(x.data.nelement() for x in GPT2_LM.parameters())
    logging('total parameters: {}'.format(total_params))
    logging('total parameters (exclude duplication): {}'.format(print_parameters(GPT2_LM, ["lm_head","weight_global",'MLP_linear_l2'])))
    test_loss_all, spend_time_arr = only_test_time_loss(dataloader_test, GPT2_LM, parallel_GPT2_LM, args)
    logging('-' * 89)
    logging('loss {} | average time {} | time std {}\n'.format(test_loss_all, np.mean(spend_time_arr), np.std(spend_time_arr)))
    sys.exit()
    
elif args.simple_eval:
    assert not args.template_experiment
    total_params = sum(x.data.nelement() for x in GPT2_LM.parameters())
    logging('total parameters: {}'.format(total_params))
    logging('total parameters (exclude duplication): {}'.format(print_parameters(GPT2_LM, ["lm_head","weight_global",'MLP_linear_l2'])))
    test_loss_all, test_div, test_count_best_arr, test_weights_arr, test_emb_div_arr_vis, test_loss_seq_vis, test_MRR_seq_vis = simple_eval(dataloader_test, GPT2_LM, parallel_GPT2_LM, n_facet_effective, args)
    logging('-' * 89)
    logging('| div {:5.3f} | weights {} | prob_best_arr {} | valid loss {:5.5f} | loss_seq_vis {} | MRR_seq_vis {} | emb_div_seq_vis {} \n '
            .format(test_div, test_weights_arr, test_count_best_arr / np.sum(test_count_best_arr), test_loss_all, test_loss_seq_vis, test_MRR_seq_vis, test_emb_div_arr_vis ))

    sys.exit()

elif args.only_eval:
    assert not args.template_experiment
    #val_loss_all, val_div, val_count_arr, val_count_best_arr, val_emb_div_arr_vis, val_loss_seq_vis, val_MRR_seq_vis, val_loss_period_sum, val_loss_period_count, val_MRR_period_sum = comprehensive_evaluate(dataloader_val)
    val_loss_all, val_div, val_count_best_arr, val_weights_arr, val_emb_div_arr_vis, val_loss_seq_vis, val_MRR_seq_vis, val_loss_period_sum, val_loss_period_count, val_MRR_period_sum = comprehensive_evaluate(dataloader_val, GPT2_LM, parallel_GPT2_LM, n_facet_effective, args, device)
    logging('-' * 89)
    logging('| div {:5.3f} | weights {} | prob_best_arr {} | valid loss {:5.5f} | loss_seq_vis {} | MRR_seq_vis {} | emb_div_seq_vis {} | \n val_loss_period_mean {} | \n val_loss_period_count {} | \n val_MRR_period_mean {} |'
            #.format(val_div, val_count_arr / np.sum(val_count_arr), val_count_best_arr / np.sum(val_count_best_arr), val_loss_all, val_loss_seq_vis, val_MRR_seq_vis, val_emb_div_arr_vis, val_loss_period_sum / (1e-15+val_loss_period_count), val_loss_period_count, val_MRR_period_sum / (1e-15+val_loss_period_count) ))
            .format(val_div, val_weights_arr, val_count_best_arr / np.sum(val_count_best_arr), val_loss_all, val_loss_seq_vis, val_MRR_seq_vis, val_emb_div_arr_vis, val_loss_period_sum / (1e-15+val_loss_period_count), val_loss_period_count, val_MRR_period_sum / (1e-15+val_loss_period_count) ))
    sys.exit()


if args.only_visualize:
    vis_file_name = os.path.join(args.save, 'visualize.txt')
    with open(vis_file_name, 'w') as f_out:
        only_visualize(dataloader_val, f_out)
    sys.exit()

if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(GPT2_LM.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(GPT2_LM.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'AdamW':
    optimizer = AdamW(GPT2_LM.parameters(), lr=args.lr, weight_decay=args.wdecay)

#if args.continue_train and not args.template_experiment:
if args.continue_train: #and not args.template_experiment:
    #optimizer_state_dict = torch.load(os.path.join(args.save, 'optimizer.pt'), map_location=device)
    optimizer_state_dict = torch.load(os.path.join(args.save, 'optimizer.pt'), map_location='cpu')
    optimizer.load_state_dict(optimizer_state_dict)
    del optimizer_state_dict

best_val_loss = None
saving_freq = int(math.floor(args.training_split_num / args.valid_per_epoch))

#after_n_valid = 0

f_out_best = None
f_out_epochs = None
if args.template_experiment:
    tokenizer_GPT2 = GPT2Tokenizer.from_pretrained(model_name)
    if len(args.reconstruction_file_name) > 0:
        f_out_best = open(args.reconstruction_file_name, 'w')

    if len(args.template_vis_file_name) > 0:
        f_out_epochs = open(args.template_vis_file_name, 'w')

training_loss_list = []
validation_loss_list = []
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    for i in range(len(dataloader_train_arr)):
        if epoch == 1 and i < args.start_training_split:
            print("Skipping epoch "+str(epoch) + ' split '+str(i) )
            continue
        training_loss = train_one_epoch(dataloader_train_arr[i], i)
        # training_loss = 0
        training_loss_list.append(training_loss)

        if i != args.training_split_num - 1 and (i + 1) % saving_freq != 0:
            continue

        str_out = None
        if (len(args.reconstruction_file_name) > 0 or len(args.template_vis_file_name) > 0) and args.template_experiment:
            str_out = io.StringIO()
        #val_loss_all, val_div, val_count_arr, val_count_best_arr = evaluate(dataloader_val)
        val_loss_all, val_div, val_count_best_arr, val_weights_arr, str_out = evaluate(dataloader_val, str_out)
        validation_loss_list.append(val_loss_all)
        logging('-' * 89)
        logging('| end of epoch {:3d} split {:3d} | time: {:5.2f}s | div {:5.3f} | weights {} | prob_best_arr {} | valid loss {:5.5f} '
                .format(epoch, i, (time.time() - epoch_start_time), val_div, val_weights_arr, val_count_best_arr / np.sum(val_count_best_arr), val_loss_all))
        
        test_loss_list = []
        for j, dataloader_test in enumerate(dataloader_test_arr):
            test_loss_all, test_div, test_count_best_arr, test_weights_arr, str_out_test = evaluate(dataloader_test, str_out = None)
            logging('| test file {} | time: {:5.2f}s | div {:5.3f} | weights {} | prob_best_arr {} | test loss {:5.5f} '
                    .format(testing_file_names_list[j], (time.time() - epoch_start_time), test_div, test_weights_arr, test_count_best_arr / np.sum(test_count_best_arr), test_loss_all))
            test_loss_list.append(test_loss_all)
            

        val_loss_important = val_loss_all

        if (best_val_loss is None or val_loss_important < best_val_loss) and str_out is not None and f_out_best is not None:
            f_out_best.write('epoch: {}, best validation loss {}\n'.format(epoch, val_loss_important))
            for j, test_loss in enumerate(test_loss_list):
                f_out_best.write('test file {}, test loss {}\n'.format(testing_file_names_list[j],test_loss))
            f_out_best.write(str_out.getvalue())
            best_val_loss = val_loss_important
            f_out_best.flush()
            logging('Update prediction file')
        #if after_n_valid % args.save_every_n_valid == 0 and str_out is not None and f_out_epochs is not None:
        if i % args.save_every_n_valid == 0 and str_out is not None and f_out_epochs is not None:
            f_out_epochs.write('epoch: {}, validation loss {}\n'.format(epoch, val_loss_important))
            for j, test_loss in enumerate(test_loss_list):
                f_out_epochs.write('test file {}, test loss {}\n'.format(testing_file_names_list[j],test_loss))
            f_out_epochs.write(str_out.getvalue())
            f_out_epochs.flush()
            

        if not args.template_experiment and (not best_val_loss or val_loss_important < best_val_loss):
            #save_checkpoint(encoder, decoder, optimizer_e, optimizer_d, external_emb, args.save)
            torch.save(GPT2_LM.state_dict(), os.path.join(args.save, 'LM_weights.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.save, 'optimizer.pt'))
            best_val_loss = val_loss_important
            logging('Models Saved')
        #after_n_valid += 1
        #if not args.template_experiment and after_n_valid % args.save_every_n_valid == 0:
        if not args.template_experiment and (i+1) % args.save_every_n_valid == 0:
            torch.save(GPT2_LM.state_dict(), os.path.join(args.save, 'LM_weights_'+str(epoch)+'_'+str(i)+'.pt'))
            logging('Save extra model')

logging('Best validation loss: {}'.format(best_val_loss))
logging('validation_loss: ' + ','.join([str(x) for x in validation_loss_list]))
logging('training_loss: ' + ','.join([str(x) for x in training_loss_list]))
if f_out_best is not None:
    f_out_best.close()
if f_out_epochs is not None:
    f_out_epochs.close()
