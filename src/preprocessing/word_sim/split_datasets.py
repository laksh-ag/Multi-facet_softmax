import os
import random
#input_file = 'resources/MEN/MEN_dataset_lemma_form_full'
input_file = './resources/hypernym_benchmarks/word_sim_list'
#contain_POS = True
contain_POS = False

#output_dir = 'data/processed/similarity_MEN/split'
output_dir = 'data/processed/similarity_hyper/split'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_input(input_path):
    input_data = []
    with open(input_path) as f_in:
        for line in f_in:
            if contain_POS:
                w1, w2, score = line.rstrip().split()
                w1_w, w1_pos = w1.split('-')
                w2_w, w2_pos = w2.split('-')
                if w1_pos != 'n' or w2_pos != 'n':
                    continue
            else:
                w1_w, w2_w, score = line.rstrip().split()
            input_data.append([w1_w, w2_w])
    return input_data

def split_data(output_data, output_prefix):
    train_size = int(len(output_data)/3)
    random.shuffle(output_data)
    with open(output_prefix+'train', 'w') as f_out:
        f_out.write('\n'.join([w1+' '+w2 for w1, w2 in output_data[:train_size] ]) )
    with open(output_prefix+'one_overlap', 'w') as f_out:
        f_out.write('\n'.join([w1+' '+w2 for w1, w2 in output_data[train_size:2*train_size] ]) )
    with open(output_prefix+'no_overlap', 'w') as f_out:
        f_out.write('\n'.join([w1+' '+w2 for w1, w2 in output_data[2*train_size:] ]) )


input_data = load_input(input_file)
print("dataset size: {}".format(len(input_data)) )
high_sim_size = int(len(input_data)/2)
high_sim_data = input_data[:high_sim_size]
low_sim_data = input_data[high_sim_size:]

split_data(high_sim_data, os.path.join(output_dir, "high_sim_"))
split_data(low_sim_data, os.path.join(output_dir, "low_sim_"))



