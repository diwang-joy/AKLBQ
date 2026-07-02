import argparse
import torch
import numpy as np
import copy

def cal_KLDiv(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))

def _smooth_distribution(dist, eps=0.0001):
    return dist + eps

def find_outlier(maxq, f2, bins):
    if f2.is_cuda:
        f2 = f2.cpu()

    counts, bin_edges = np.histogram(f2.numpy(), bins=bins, density=False, range=None)
    intensive_counts = np.argmax(counts)

    counts_left = counts[:intensive_counts][::-1]
    counts_right = counts[intensive_counts:]
    bin_left = bin_edges[:intensive_counts]
    bin_right = bin_edges[intensive_counts:]

    threshold = bins
    target_bin_left = int(maxq * bin_left.size // threshold)
    target_bin_left = max(target_bin_left, 1)

    kl_array_left = np.zeros(counts_left.size - target_bin_left - 1)

    for i in range(target_bin_left + 1, counts_left.size): 
        sliced_arr_left = copy.deepcopy(counts_left[:i])
        sliced_arr_left[i-1] += sum(counts_left[i:])
        is_nonzeros_left = (sliced_arr_left != 0).astype(np.int64)
        quantized_bins_left = np.zeros(target_bin_left, dtype=np.float32)
        num_merged_bins_left = sliced_arr_left.size // target_bin_left

        for j in range(target_bin_left):
            start, stop = j * num_merged_bins_left, (j+1) * num_merged_bins_left
            quantized_bins_left[j] = sliced_arr_left[start:stop].sum()
        quantized_bins_left[-1] += sliced_arr_left[target_bin_left * num_merged_bins_left:].sum()

        f2_q_left = np.zeros(sliced_arr_left.size, dtype=np.float64)
        for j in range(target_bin_left):
            start, stop = j * num_merged_bins_left, (j+1) * num_merged_bins_left if j < target_bin_left - 1 else f2_q_left.size
            norm = is_nonzeros_left[start:stop].sum()
            if norm != 0:
                mask = (sliced_arr_left[start:stop] != 0)
                f2_q_left[start:stop] = (quantized_bins_left[j] / norm) * mask

        f2_q_left = _smooth_distribution(f2_q_left)
        sliced_arr_left = _smooth_distribution(sliced_arr_left)
        kl_array_left[i - target_bin_left - 1] = cal_KLDiv(sliced_arr_left, f2_q_left)

    kl_array_left_min = np.argmin(kl_array_left)
    target_bin_right = maxq - target_bin_left
    bin_left = torch.tensor(bin_left, dtype=torch.float32)
    bin_right = torch.tensor(bin_right, dtype=torch.float32)
    target_bin_min_left = bin_left[bin_left.size(0) - (kl_array_left_min + target_bin_left + 1)]

    return bin_right[-1], target_bin_min_left

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust the number of bins for outlier detection.")
    parser.add_argument("--bins", type=int, default=64, help="Number of bins for histogram")
    parser.add_argument("--maxq", type=int, default=128, help="Max quantization level")
    args = parser.parse_args()

    # 生成随机输入数据
    f2 = torch.randn(1000)

    # 运行 find_outlier
    right, left = find_outlier(args.maxq, f2, args.bins)
    print(f"Right Threshold: {right}, Left Threshold: {left}")
