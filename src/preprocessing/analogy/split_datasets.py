
import os
import random

#input_dir = 'resources/question-data/'
input_dir = 'resources/google_analogy_split'
people_files = ['family.txt']
location_files = ['capital-common-countries.txt', 'capital-world.txt', 'city-in-state.txt', 'three_locations.txt']

output_dir = 'data/processed/analogy_google/split'

output_people_prefix = 'people_'
output_location_prefix = 'location_'

dataset_percentage=[0.1, 0.5]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_input(input_path):
    input_data = []
    w_pair_set = set()
    with open(input_path) as f_in:
        for line in f_in:
            w1, w2, w3, w4 = line.rstrip().split()
            input_data.append([w1, w2, w3, w4])
            w_pair_set.add( (w1, w2) )
            w_pair_set.add( (w3, w4) )
    return input_data, w_pair_set

def split_data(input_data, w_pair_set,  percent):
    num_pair = len(w_pair_set)
    train_pair_num = int( num_pair * percent )
    assert train_pair_num > 0, print(num_pair, percent)
    w_pair_list = list(w_pair_set)
    random.shuffle(w_pair_list)
    train_pair_set = set(w_pair_list[:train_pair_num])
    training_data = []
    one_overlap_data = []
    no_overlap_data = []
    for i in range(len(input_data)):
        w1, w2, w3, w4 = input_data[i]
        num_overlap = 0
        if (w1, w2) in train_pair_set:
            num_overlap += 1
        if (w3, w4) in train_pair_set:
            num_overlap += 1
        if num_overlap == 0:
            no_overlap_data.append([w1, w2, w3, w4])
        elif num_overlap == 1:
            one_overlap_data.append([w1, w2, w3, w4])
        elif num_overlap == 2:
            training_data.append([w1, w2, w3, w4])

    print(len(training_data), len(one_overlap_data), len(no_overlap_data))
    assert len(training_data) > 0, train_pair_set
    return training_data, one_overlap_data, no_overlap_data

def store_output(store_data, output_file_name):
    with open(output_file_name, 'w') as f_out:
        for fields in store_data:
            f_out.write(' '.join(fields)+'\n')
    

def create_splits(input_files, output_prefix):
    for percent in dataset_percentage:
        print("percentage {}".format(percent))
        output_folder = os.path.join(output_dir, str(percent))
        if not os.path.exists( output_folder ):
            os.makedirs(output_folder)
        for f_name in input_files:
            print(f_name)
            input_data, w_pair_set = load_input(os.path.join(input_dir, f_name))
            training_data, one_overlap_data, no_overlap_data = split_data(input_data, w_pair_set, percent)
            store_output(training_data, os.path.join(output_folder, output_prefix + f_name.replace('.txt','') + '_train') )
            store_output(one_overlap_data, os.path.join(output_folder, output_prefix + f_name.replace('.txt','') + '_one_overlap') )
            store_output(no_overlap_data, os.path.join(output_folder, output_prefix + f_name.replace('.txt','') + '_no_overlap') )

create_splits(people_files, output_people_prefix)
create_splits(location_files, output_location_prefix)
