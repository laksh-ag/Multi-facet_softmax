import torch

#input_dir = "/iesl/canvas/hschang/language_modeling/interactive_LM/data/processed/wiki2016_gpt2/tensors_all_min100/"
input_dir = "/iesl/canvas/hschang/language_modeling/interactive_LM/data/processed/wiki2016_gpt2/tensors_100000_min100/"
#output_dir = './data/processed/wiki2016_gpt2/tensors_all_min100/'
output_dir = './data/processed/wiki2016_gpt2/tensors_100000_min100/'
proc_file_name = 'train.pt'
#proc_file_name = 'val_org.pt'
#proc_file_name = 'test_org.pt'
input_tensor_file = input_dir + proc_file_name
output_tensor_file = output_dir + proc_file_name

with open(input_tensor_file, 'rb') as f_in:
    w_ind_gpt2_tensor, w_ind_spacy_tensor, idx_gpt2_to_spacy_tensor = torch.load(f_in, map_location='cpu')
torch.save(w_ind_gpt2_tensor, output_tensor_file)
