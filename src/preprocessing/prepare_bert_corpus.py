import random

input_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/"
input_file = input_dir + "wiki2016_both.txt"
#output_train_file = input_dir + "wiki2016_org_train_small.txt"
#output_val_file = input_dir + "wiki2016_org_val_small.txt"
output_train_file = input_dir + "wiki2016_org_train.txt"
output_val_file = input_dir + "wiki2016_org_val.txt"

training_ratio = 0.9

max_line_num = 1000000000000
#max_line_num = 100000

output_arr = []

with open(input_file) as f_in:
    for i, line in enumerate(f_in):
        field = line.rstrip().split('\t')
        output_arr.append(" ".join(field[0].split()))
        if i > max_line_num:
            break

idx_shuffled = list(range(len(output_arr)))
random.shuffle(idx_shuffled)
training_size = int(len(output_arr)*training_ratio)

with open(output_train_file,'w') as f_out:
    f_out.write('\n'.join([output_arr[i] for i in idx_shuffled[:training_size]]))

with open(output_val_file,'w') as f_out:
    f_out.write( '\n'.join([output_arr[i] for i in idx_shuffled[training_size:]]))
