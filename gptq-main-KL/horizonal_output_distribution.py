import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from scipy import stats
import copy

# model_original=torch.load('original_model')
# model_8bit=torch.load('8bit_model')
# model_4bit=torch.load('4bit_model')
# model_3bit=torch.load('3bit_model')
current_path = os.getcwd()
print(current_path)
DEV = torch.device('cuda:0')



############  horizonal models  weights of layer 0 K output start ###############
# new_dir = "/home/zhang/code2/gptq-main/4 models horizonal_distribution/layer 0/4 models layer 0 k"
# os.chdir(new_dir)
# current_path = os.getcwd()
# print(current_path)

# f = model_original['model.decoder.layers.0.self_attn.k_proj.weight'].flatten()
# plt.title("original_model.layers.0.self_attn.k_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('orginal_model layer 0 self_attn.k_proj.png') 


# f = model_8bit['model.decoder.layers.0.self_attn.k_proj.weight'].flatten()
# plt.title("8bit_model.layers.0.self_attn.k_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('8bit_model layer 0 self_attn.k_proj.png') 

# f = model_4bit['model.decoder.layers.0.self_attn.k_proj.weight'].flatten()
# plt.title("4bit_model.layers.0.self_attn.k_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('4bit_model layer 0 self_attn.k_proj.png') 

# f = model_3bit['model.decoder.layers.0.self_attn.k_proj.weight'].flatten()
# plt.title("3bit_model.layers.0.self_attn.k_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('3bit_model layer 0 self_attn.k_proj.png') 
############  horizonal models  weights of layer 0 K output end ###############

############  horizonal models  weights of layer 0 Q output start ###############
# new_dir = "/home/zhang/code2/gptq-main/4 models horizonal_distribution/layer 0/4 models layer 0 q"
# os.chdir(new_dir)
# current_path = os.getcwd()
# print(current_path)

# f = model_original['model.decoder.layers.0.self_attn.q_proj.weight'].flatten()
# plt.title("original_model.layers.0.self_attn.q_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('orginal_model layer 0 self_attn.q_proj.png') 


# f = model_8bit['model.decoder.layers.0.self_attn.q_proj.weight'].flatten()
# plt.title("8bit_model.layers.0.self_attn.q_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('8bit_model layer 0 self_attn.q_proj.png') 

# f = model_4bit['model.decoder.layers.0.self_attn.q_proj.weight'].flatten()
# plt.title("4bit_model.layers.0.self_attn.q_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('4bit_model layer 0 self_attn.q_proj.png') 

# f = model_3bit['model.decoder.layers.0.self_attn.q_proj.weight'].flatten()
# plt.title("3bit_model.layers.0.self_attn.q_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('3bit_model layer 0 self_attn.q_proj.png') 
############  horizonal models  weights of layer 0 Q output end ###############

############  horizonal models  weights of layer 0 V output start ###############
# new_dir = "/home/zhang/code2/gptq-main/4 models horizonal_distribution/layer 0/4 models layer 0 v"
# os.chdir(new_dir)
# current_path = os.getcwd()
# print(current_path)

# f = model_original['model.decoder.layers.0.self_attn.v_proj.weight'].flatten()
# plt.title("original_model.layers.0.self_attn.v_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('orginal_model layer 0 self_attn.v_proj.png') 


# f = model_8bit['model.decoder.layers.0.self_attn.v_proj.weight'].flatten()
# plt.title("8bit_model.layers.0.self_attn.v_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('8bit_model layer 0 self_attn.v_proj.png') 

# f = model_4bit['model.decoder.layers.0.self_attn.v_proj.weight'].flatten()
# plt.title("4bit_model.layers.0.self_attn.v_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('4bit_model layer 0 self_attn.v_proj.png') 

# f = model_3bit['model.decoder.layers.0.self_attn.v_proj.weight'].flatten()
# plt.title("3bit_model.layers.0.self_attn.v_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('3bit_model layer 0 self_attn.v_proj.png') 
############  horizonal models  weights of layer 0 V output end ###############

############  horizonal models  weights of layer 0 OUT output start ###############
# new_dir = "/home/zhang/code2/gptq-main/4 models horizonal_distribution/layer 0/4 models layer 0 out"
# os.chdir(new_dir)
# current_path = os.getcwd()
# print(current_path)

# f = model_original['model.decoder.layers.0.self_attn.out_proj.weight'].flatten()
# plt.title("original_model.layers.0.self_attn.out_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('orginal_model layer 0 self_attn.out_proj.png') 


# f = model_8bit['model.decoder.layers.0.self_attn.out_proj.weight'].flatten()
# plt.title("8bit_model.layers.0.self_attn.out_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('8bit_model layer 0 self_attn.out_proj.png') 

# f = model_4bit['model.decoder.layers.0.self_attn.out_proj.weight'].flatten()
# plt.title("4bit_model.layers.0.self_attn.out_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('4bit_model layer 0 self_attn.out_proj.png') 

# f = model_3bit['model.decoder.layers.0.self_attn.out_proj.weight'].flatten()
# plt.title("3bit_model.layers.0.self_attn.out_proj.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('3bit_model layer 0 self_attn.out_proj.png') 
############  horizonal models  weights of layer 0 OUT output end ###############

############  horizonal models  weights of layer 0 FC1 output start ###############
# new_dir = "/home/zhang/code2/gptq-main/4 models horizonal_distribution/layer 0/4 models layer 0 fc1"
# os.chdir(new_dir)
# current_path = os.getcwd()
# print(current_path)

# f = model_original['model.decoder.layers.0.fc1.weight'].flatten()
# plt.title("original_model.layers.0.fc1.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('orginal_model layer 0 fc1.png') 


# f = model_8bit['model.decoder.layers.0.fc1.weight'].flatten()
# plt.title("8bit_model.layers.0.fc1.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('8bit_model layer 0 fc1.png') 

# f = model_4bit['model.decoder.layers.0.fc1.weight'].flatten()
# plt.title("4bit_model.layers.0.fc1.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('4bit_model layer 0 fc1.png') 

# f = model_3bit['model.decoder.layers.0.fc1.weight'].flatten()
# plt.title("3bit_model.layers.0.fc1.weight")
# plt.hist(f, bins = 100, density=True, histtype='step')
# plt.savefig('3bit_model layer 0 fc1.png') 
############  horizonal models  weights of layer 0 FC1 output end ###############

def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor 
    and taking the corresponding amount off the non-zero values.
    Ref: http://web.engr.illinois.edu/~hanj/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    
    return hist

def cal_KLDiv(p, q):
    p = np.array(p, dtype=np.float32)
    q = np.array(q, dtype=np.float32)

    assert p.shape == q.shape, "Distributions must be the same shape"
    
    # Normalize p and q to make them probability distributions
    p_sum = np.sum(p)
    q_sum = np.sum(q)
    if p_sum == 0 or q_sum == 0:
        raise ValueError("Sum of distribution cannot be zero")

    p = p / p_sum
    q = q / q_sum

    # Compute KL divergence
    # Equivalent to: sum(p[i] * log(p[i] / q[i]))
    mask = (p > 0) & (q > 0)
    if not np.all(mask):
        raise ValueError("p and q must be strictly positive in all bins for KL computation")

    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


############  horizonal models  weights of layer 0 FC2 output start ###############
# new_dir = "/home/wangdi/code1/gptq-main/4 models horizonal_distribution/layer 0/4 models layer 0 fc2"
# os.chdir(new_dir)
# current_path = os.getcwd()
# print(current_path)

# f1 = model_original['model.decoder.layers.0.fc2.weight'].flatten()
# counts, bin_edges = np.histogram(f1, bins=128)
# # print(bin_edges)
# plt.title("original_model.layers.0.fc2.weight")
# plt.hist(f1, bins = 2048, density=False, histtype='step')
# plt.savefig('orginal_model layer 0 fc2.png') 
# f2 = model_8bit['model.decoder.layers.0.fc2.weight']

def find_outlier(maxq,f2,bins):
    f2 = f2.flatten()
    # 如果 f2 是 GPU 张量，先转到 CPU
    if f2.is_cuda:
         f2 = f2.cpu()
    counts, bin_edges = np.histogram(f2.numpy(), bins=bins, density=False, range=None)
    threshold = bins
    target_bin = maxq
    kl_array = np.zeros(int((threshold - target_bin)/2))
    for i in range(target_bin+2, threshold, 2): 
        mid = int(threshold/2)

        mid_arr = copy.deepcopy(counts[mid - int(i/2):mid+int(i/2)+1])
        start = mid - int(i/2)
        stop = mid+int(i/2)+1
        for j in range(0, threshold):
            if(j< start):
                mid_arr[0] += counts[j]
            elif(j>= stop):
                mid_arr[i-1] += counts[j]
        is_nonzeros = (mid_arr != 0).astype(np.int64)
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        num_merged_bins = mid_arr.size // target_bin
        #merge
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = mid_arr[start:stop].sum()
        quantized_bins[-1] += mid_arr[target_bin * num_merged_bins:].sum()
        #expand
        f2_q = np.zeros(mid_arr.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = f2_q.size
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                mask = (mid_arr[start:stop] != 0) 
                f2_q[start:stop] = (float(quantized_bins[j]) / float(norm)) * mask
        mid_arr = _smooth_distribution(mid_arr,eps = 0.0001)
        f2_q = _smooth_distribution(f2_q,eps=0.0001)
        kl_array[int((threshold - i)/2)] = cal_KLDiv(mid_arr, f2_q)


    kl_array = kl_array[::-1]
    kl_array_min = np.argmin(kl_array)
    real_bins =  kl_array_min * 2 + target_bin + 2
    bin_right_outlier = bin_edges[int(threshold/2) + int(real_bins/2) + 1]
    bin_right_outlier = torch.tensor(bin_right_outlier, dtype=torch.float32)
    middle_edges = bin_edges[32]
    # if(bin_right_outlier < 0):
    #      return bin_right_outlier, middle_edges-(bin_right_outlier-middle_edges)
    # else:
    return bin_right_outlier, -bin_right_outlier







