import torch
import os

input_folder = "data/processed/analogy_google/tensor_more/0.5/"
output_folder = "data/processed/analogy_google/tensor_more/0.5/"
#input_folder = "data/processed/analogy_google/tensor_more_w1w3/0.5/"
#output_folder = "data/processed/analogy_google/tensor_more_w1w3/0.5/"
#input_folder = "data/processed/analogy_google/tensor_w1w3/0.5/"
#output_folder = "data/processed/analogy_google/tensor_w1w3/0.5/"
#input_folder = "data/processed/analogy_google/tensor/0.5/"
#output_folder = "data/processed/analogy_google/tensor/0.5/"

input_1_name = "location_three_locations"
input_2_name = "people_family"
output_name = "all_four"

subset_suffix_arr = ["_train", "_no_overlap", "_one_overlap"]

for suffix in subset_suffix_arr:
    sent_tensor_1, sent_len_tensor_1, target_position_tensor_1 = torch.load( os.path.join(input_folder, input_1_name+suffix) )
    sent_tensor_2, sent_len_tensor_2, target_position_tensor_2 = torch.load( os.path.join(input_folder, input_2_name+suffix) )
    num_rows_1 = sent_tensor_1.size(0)
    num_rows_2 = sent_tensor_2.size(0)
    max_sent_len_1 = sent_tensor_1.size(1)
    max_sent_len_2 = sent_tensor_2.size(1)
    max_sent_len_12 = max(max_sent_len_1, max_sent_len_2)
    if max_sent_len_1 < max_sent_len_12:
        sent_tensor_1 = torch.hstack( [sent_tensor_1, torch.zeros( (num_rows_1, max_sent_len_12-max_sent_len_1) ) ] )
    elif max_sent_len_2 < max_sent_len_12:
        sent_tensor_2 = torch.hstack( [sent_tensor_2, torch.zeros( (num_rows_2, max_sent_len_12-max_sent_len_2) ) ] )
    sent_tensor_12 = torch.cat([sent_tensor_1, sent_tensor_2], dim=0)
    sent_len_tensor_12 = torch.cat([sent_len_tensor_1, sent_len_tensor_2], dim=0)
    target_position_tensor_12 = torch.cat([target_position_tensor_1, target_position_tensor_2], dim=0)
    print(sent_tensor_12.size())
    print(sent_len_tensor_12.size())
    print(target_position_tensor_12.size())
    torch.save([sent_tensor_12, sent_len_tensor_12, target_position_tensor_12], os.path.join(output_folder, output_name+suffix))
