import os
import argparse
import numpy as np
from sklearn.externals import joblib


def writeNonZeroEntry(prep_fn, tensor_folder, tensor_output_path, slab_inds, vocab_size):
    """
    transform each slab to lines for tensor decomposition
    """
    """
    f = open(prep_fn, "r")
    lines = f.readlines()
    f.close()
    slab_num = len(lines)
    """
    g = open(tensor_output_path, "a+")
    for slab_ind in slab_inds:
        with open(tensor_folder+str(slab_ind)+".sav", "rb") as handle:
            count_mat = joblib.load(handle)
        # only get most frequent vocab
        print "count_mat original shape:", count_mat.shape,
        count_mat = count_mat[:vocab_size, :vocab_size]
        print "count_mat is reduced to:", count_mat.shape

        nonzero_inds = np.nonzero(count_mat)
        nonzero_values = count_mat[nonzero_inds]
        nonzero_coords = np.transpose(nonzero_inds) + 1
        for ind in range(len(nonzero_values)):
	    val = nonzero_values[ind]
            #log_value = np.log(nonzero_values[ind]+1)
            seq = [str(v) for v in nonzero_coords[ind]] + [str(slab_ind+1)] + [str(val)]
            print >> g, " ".join(seq)
    g.close()
    print "done formatting", slab_inds


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepNum', default=49, type=int)
    args = parser.parse_args()
    prep_num = args.prepNum
    
    # data path
    data_folder = "../../data/"
    prep_fn = data_folder + "prepositions_" + str(prep_num) + ".txt"
    slab_inds = range(prep_num + 1) 
    tensor_folder = "../../model/"
    tensor_output_path = tensor_folder+"tensor_" + str(prep_num) + ".txt"
    
    vocab_size = 50000
    writeNonZeroEntry(prep_fn, tensor_folder, tensor_output_path, slab_inds, vocab_size)
