
import os

input_file = 'resources/google_analogy'
output_dir = 'resources/google_analogy_split'

f_out = None
with open(input_file) as f_in:
    for line in f_in:
        content = line.rstrip()
        if content[:2] == ': ':
            file_name = content[2:]
            if f_out is not None:
                f_out.close()
            f_out = open(os.path.join(output_dir, file_name + '.txt'), 'w')
            continue
        else:
            f_out.write(content+'\n')
        
            
