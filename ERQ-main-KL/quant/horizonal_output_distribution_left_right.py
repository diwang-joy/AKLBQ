import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from scipy import stats
import copy

DEV = torch.device('cuda:0')





def _smooth_distribution(p, eps=0.0001):
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

def cal_KLDiv(tensor_3bit, tensor_8bit):

    output = stats.entropy(tensor_3bit, tensor_8bit)
    return output


def find_outlier(maxq,f2,bins):
    f2 = f2.flatten()
    # 如果 f2 是 GPU 张量，先转到 CPU
    if f2.is_cuda:
         f2 = f2.cpu()
    #对传入的数据进行直方分布处理，并统计每个bin内的数据分布数量
    counts, bin_edges = np.histogram(f2.numpy(), bins=bins, density=False, range=None)
    
    #找出count数最大的位置，即数据分布最密集的区域
    intensive_counts = np.argmax(counts)

    #将数据分为左右两个部分，方便后续的双向遍历找出KL散度最小的位置
    counts_left = counts[:intensive_counts]
    counts_right = counts[intensive_counts:]
    #将左半部分数据进行翻转，以按照bin[0,1,2,...]顺序输出KL散度
    counts_left = counts_left[::-1]
    #从数据最密集区域向左右取出边界值
    bin_left = bin_edges[:intensive_counts]
    bin_right = bin_edges[intensive_counts:]
    #设置target_bin,即裁减区域
    threshold = bins
    target_bin_left = int(maxq * bin_left.size // threshold)
    target_bin_right = maxq -target_bin_left
    #
    if target_bin_left == 0:
        target_bin_left = 1
        target_bin_right = target_bin_right - 1
    # 

    if counts_left.size > 2 and counts_right.size > 2:


        #left
        kl_array_left = np.zeros(int(counts_left.size-target_bin_left)-1)
        for i in range(target_bin_left+1, counts_left.size): 
            sliced_arr_left = copy.deepcopy(counts_left[:i])

            sliced_arr_left[i-1] += sum(counts_left[i:])
            is_nonzeros_left = (sliced_arr_left != 0).astype(np.int64) 
            quantized_bins_left = np.zeros(target_bin_left, dtype=np.float32)
            num_merged_bins_left = sliced_arr_left.size // target_bin_left
            #merge
            for j in range(target_bin_left):
                start = j * num_merged_bins_left
                stop = start + num_merged_bins_left
                quantized_bins_left[j] = sliced_arr_left[start:stop].sum()
            quantized_bins_left[-1] += sliced_arr_left[target_bin_left * num_merged_bins_left:].sum()
            #expand
            f2_q_left = np.zeros(sliced_arr_left.size, dtype=np.float64)
            for j in range(target_bin_left):
                start = j * num_merged_bins_left
                if j == target_bin_left - 1:
                    stop = f2_q_left.size
                else:
                    stop = start + num_merged_bins_left
                norm = is_nonzeros_left[start:stop].sum()
                if norm != 0:
                    mask = (sliced_arr_left[start:stop] != 0) 
                    f2_q_left[start:stop] = (float(quantized_bins_left[j]) / float(norm)) * mask
            f2_q_left = _smooth_distribution(f2_q_left,eps=0.0001)
            sliced_arr_left = _smooth_distribution(sliced_arr_left,eps=0.0001)
            kl_array_left[i-target_bin_left-1] = cal_KLDiv(sliced_arr_left, f2_q_left)
        kl_array_left_min = np.argmin(kl_array_left)



        #right
        kl_array_right = np.zeros(int(counts_right.size-target_bin_right)-1)
        for i in range(target_bin_right+1, counts_right.size): 
            sliced_arr_right = copy.deepcopy(counts_right[:i])

            sliced_arr_right[i-1] += sum(counts_right[i:])

            is_nonzeros_right = (sliced_arr_right != 0).astype(np.int64) 
            quantized_bins_right = np.zeros(target_bin_right, dtype=np.float32) 
            num_merged_bins_right = sliced_arr_right.size // target_bin_right
            #merge
            for j in range(target_bin_right):
                start = j * num_merged_bins_right
                stop = start + num_merged_bins_right
                quantized_bins_right[j] = sliced_arr_right[start:stop].sum()
            quantized_bins_right[-1] += sliced_arr_right[target_bin_right * num_merged_bins_right:].sum()
            #expand
            f2_q_right = np.zeros(sliced_arr_right.size, dtype=np.float64)
            for j in range(target_bin_right):
                start = j * num_merged_bins_right
                if j == target_bin_right - 1:
                    stop = f2_q_right.size
                else:
                    stop = start + num_merged_bins_right
                norm = is_nonzeros_right[start:stop].sum()
                if norm != 0:
                    mask = (sliced_arr_right[start:stop] != 0) 
                    f2_q_right[start:stop] = (float(quantized_bins_right[j]) / float(norm)) * mask
            f2_q_right = _smooth_distribution(f2_q_right,eps=0.0001)
            sliced_arr_right = _smooth_distribution(sliced_arr_right,eps=0.0001)
            # f2_probability_q = counts_new/sum(counts_new)
            kl_array_right[i-target_bin_right-1] = cal_KLDiv(sliced_arr_right, f2_q_right)

        kl_array_right_min = np.argmin(kl_array_right)
        bin_left = torch.tensor(bin_left, dtype=torch.float32)
        bin_right = torch.tensor(bin_right, dtype=torch.float32)
        target_bin_min_right = bin_right[kl_array_right_min+target_bin_right+2]
        target_bin_min_left = bin_left[bin_left.size(0)-(kl_array_left_min+target_bin_left+3)] 
        return target_bin_min_right,target_bin_min_left
    elif counts_left.size <= 2:
        kl_array_right = np.zeros(int(counts_right.size-target_bin_right)-1)
        #right
        for i in range(target_bin_right+1, counts_right.size): 
            sliced_arr_right = copy.deepcopy(counts_right[:i])

            sliced_arr_right[i-1] += sum(counts_right[i:])

            is_nonzeros_right = (sliced_arr_right != 0).astype(np.int64) 
            quantized_bins_right = np.zeros(target_bin_right, dtype=np.float32) 
            num_merged_bins_right = sliced_arr_right.size // target_bin_right
            #merge
            for j in range(target_bin_right):
                start = j * num_merged_bins_right
                stop = start + num_merged_bins_right
                quantized_bins_right[j] = sliced_arr_right[start:stop].sum()
            quantized_bins_right[-1] += sliced_arr_right[target_bin_right * num_merged_bins_right:].sum()
            #expand
            f2_q_right = np.zeros(sliced_arr_right.size, dtype=np.float64)
            for j in range(target_bin_right):
                start = j * num_merged_bins_right
                if j == target_bin_right - 1:
                    stop = f2_q_right.size
                else:
                    stop = start + num_merged_bins_right
                norm = is_nonzeros_right[start:stop].sum()
                if norm != 0:
                    mask = (sliced_arr_right[start:stop] != 0) 
                    f2_q_right[start:stop] = (float(quantized_bins_right[j]) / float(norm)) * mask
            f2_q_right = _smooth_distribution(f2_q_right,eps=0.0001)
            sliced_arr_right = _smooth_distribution(sliced_arr_right,eps=0.0001)
            # f2_probability_q = counts_new/sum(counts_new)
            kl_array_right[i-target_bin_right-1] = cal_KLDiv(sliced_arr_right, f2_q_right)

        kl_array_right_min = np.argmin(kl_array_right)
        bin_right = torch.tensor(bin_right, dtype=torch.float32)
        target_bin_min_right = bin_right[kl_array_right_min+target_bin_right+2]
        return target_bin_min_right, bin_edges[0]
    elif counts_right.size <= 2:

        #left
        kl_array_left = np.zeros(int(counts_left.size-target_bin_left)-1)
        for i in range(target_bin_left+1, counts_left.size): 
            sliced_arr_left = copy.deepcopy(counts_left[:i])

            sliced_arr_left[i-1] += sum(counts_left[i:])
            is_nonzeros_left = (sliced_arr_left != 0).astype(np.int64) 
            quantized_bins_left = np.zeros(target_bin_left, dtype=np.float32)
            num_merged_bins_left = sliced_arr_left.size // target_bin_left
            #merge
            for j in range(target_bin_left):
                start = j * num_merged_bins_left
                stop = start + num_merged_bins_left
                quantized_bins_left[j] = sliced_arr_left[start:stop].sum()
            quantized_bins_left[-1] += sliced_arr_left[target_bin_left * num_merged_bins_left:].sum()
            #expand
            f2_q_left = np.zeros(sliced_arr_left.size, dtype=np.float64)
            for j in range(target_bin_left):
                start = j * num_merged_bins_left
                if j == target_bin_left - 1:
                    stop = f2_q_left.size
                else:
                    stop = start + num_merged_bins_left
                norm = is_nonzeros_left[start:stop].sum()
                if norm != 0:
                    mask = (sliced_arr_left[start:stop] != 0) 
                    f2_q_left[start:stop] = (float(quantized_bins_left[j]) / float(norm)) * mask
            f2_q_left = _smooth_distribution(f2_q_left,eps=0.0001)
            sliced_arr_left = _smooth_distribution(sliced_arr_left,eps=0.0001)
            kl_array_left[i-target_bin_left-1] = cal_KLDiv(sliced_arr_left, f2_q_left)
        kl_array_left_min = np.argmin(kl_array_left)

        bin_left = torch.tensor(bin_left, dtype=torch.float32)
        target_bin_min_left = bin_left[bin_left.size(0)-(kl_array_left_min+target_bin_left+3)] 
        return bin_edges[-1],target_bin_min_left

    

