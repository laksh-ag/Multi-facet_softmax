import scipy.stats
import numpy as np

#result_file_1 = "gen_log/single_facet_top10_to_1_recon_err"
#result_file_2 = "gen_log/multi_facet_top10_to_1_recon_err"
result_file_1 = "gen_log/single_facet_top3_to_1_recon_err"
result_file_2 = "gen_log/multi_facet_top3_to_1_recon_err"

def load_result_file(result_file_name):
    with open(result_file_name) as f_in:
        results_list = []
        for line in f_in:
            input_idx, loss, reconstruction_err = line.rstrip().split()
            results_list.append( [int(input_idx), float(loss), float(reconstruction_err)])
    input_idx, loss, reconstruction_err = zip(*results_list)
    return np.array(loss), np.array(reconstruction_err)

loss_1, reconstruction_err_1 = load_result_file(result_file_1)
loss_2, reconstruction_err_2 = load_result_file(result_file_2)

print(scipy.stats.pearsonr(loss_2 - loss_1, reconstruction_err_2)) #expect higher reconstruction_err imply lower loss difference, so negative correlation
print(scipy.stats.pearsonr(loss_2 - loss_1, reconstruction_err_1))

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

#fig = plt.figure()
#plt.scatter(loss_2 - loss_1, reconstruction_err_2)
#fig.savefig('temp.png')
