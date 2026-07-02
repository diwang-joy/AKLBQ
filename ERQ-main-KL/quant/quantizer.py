# import warnings
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.nn.parameter import Parameter
# from collections import Counter
# import numpy as np
# # 假设 hod 模块在您的项目中是可用的
# import quant.horizonal_output_distribution_left_right as hod
# import matplotlib


# def robust_kl_divergence(p, q):
#     """最稳健的 KL 散度计算"""
#     # 转为概率分布（使用 softmax）
#     p_softmax = torch.nn.functional.softmax(p.flatten(), dim=0)
#     q_softmax = torch.nn.functional.softmax(q.flatten(), dim=0)
    
#     # 添加极小值
#     p_softmax = p_softmax + 1e-10
#     q_softmax = q_softmax + 1e-10
    
#     # 归一化
#     p_softmax = p_softmax / p_softmax.sum()
#     q_softmax = q_softmax / q_softmax.sum()
    
#     # 计算 KL
#     kl = (p_softmax * (p_softmax.log() - q_softmax.log())).sum()
    
#     return kl.item()


# def lp_loss(pred, tgt, p=2.0, reduction='none'):
#     """
#     loss function measured in L_p Norm
#     """
#     if reduction == 'none':
#         return (pred - tgt).abs().pow(p).sum(1).mean()
#     else:
#         return (pred - tgt).abs().pow(p).mean()


# class UniformQuantizer(nn.Module):
#     """
#     PyTorch Function that can be used for asymmetric quantization (also called uniform affine
#     quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
#     through' on the backward pass, ignoring the quantization that occurred.
#     Based on https://arxiv.org/abs/1806.08342.
#     :param n_bits: number of bit for quantization of inliers
#     :param channel_wise: if True, compute scale and zero_point in each channel
#     :param is_act: if True, it is for activation quantization
#     :param is_mixed_precision: if True, enable mixed-precision quantization for activations
#     :param n_bits_outlier: number of bit for quantization of outliers
#     """

#     def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act: bool = False,
#                  two_part: bool = False, split_quantization: bool = False,
#                  # ===== 新增参数: 用于混合精度量化 =====
#                  is_mixed_precision: bool = False, n_bits_outlier: int = 4):

#         super(UniformQuantizer, self).__init__()
#         assert 2 <= n_bits <= 8, 'bitwidth not supported'
#         self.n_bits = n_bits
#         self.n_levels = 2 ** self.n_bits
#         self.delta = None
#         self.zero_point = None
#         self.register_buffer('inited', torch.zeros(1))
#         self.channel_wise = channel_wise
#         self.is_act = is_act
#         self.store_input = None
#         self.two_part = two_part
#         self.split_quantization = split_quantization
#         self.negative_mask = None

#         # ===== 混合精度新增逻辑 =====
#         self.is_mixed_precision = is_mixed_precision
#         if self.is_mixed_precision:
#             assert self.is_act, "Mixed precision is only implemented for activations."
#             self.n_bits_outlier = n_bits_outlier
#             self.n_levels_outlier = 2 ** self.n_bits_outlier
#             # 为异常值（outliers）创建新的参数
#             self.delta_outlier = None
#             self.zero_point_outlier = None
#             # 创建 buffer 来存储正常值（inliers）的范围
#             self.register_buffer('range_min', torch.zeros(1))
#             self.register_buffer('range_max', torch.zeros(1))
#         # ============================

#     def __repr__(self):
#         s = super(UniformQuantizer, self).__repr__()
#         s = "(" + s + " inited={}, channel_wise={}, is_mixed_precision={})".format(self.inited, self.channel_wise,
#                                                                                    self.is_mixed_precision)
#         return s

#     def forward(self, x: torch.Tensor, round_way='round', index=None):

#         if self.inited == 0:
#             # ===== 混合精度的新初始化逻辑 =====
#             if self.is_mixed_precision:
#                 # 1. 使用 hod.find_outlier 确定正常值范围
#                 # 注意：您代码中使用了硬编码的7和32。这里我将其改为由 n_bits 决定，更具通用性。
#                 # max_q_val = self.n_levels - 1
#                 max_q_val = 7  # 保持和您代码中一致的硬编码值
#                 self.range_max, self.range_min = hod.find_outlier(max_q_val, x.clone().detach(), 32)

#                 # 2. 为“正常值”计算 delta 和 zero_point
#                 delta_inlier = (self.range_max - self.range_min) / (self.n_levels - 1)
#                 zero_point_inlier = (-self.range_min / delta_inlier).round()
#                 self.delta = Parameter(delta_inlier)
#                 self.zero_point = Parameter(zero_point_inlier)

#                 # 3. 为“异常值”计算 delta 和 zero_point (使用整个张量的 min/max)
#                 x_clone = x.clone().detach()
#                 x_max = x_clone.max()
#                 x_min = x_clone.min()
#                 delta_outlier = (x_max - x_min) / (self.n_levels_outlier - 1)
#                 zero_point_outlier = (-x_min / delta_outlier).round()
#                 self.delta_outlier = Parameter(delta_outlier)
#                 self.zero_point_outlier = Parameter(zero_point_outlier)
#             # ====================================
#             elif not self.two_part:
#                 delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
#                 self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()
#             else:  # two_part logic (remains unchanged)
#                 indices_list = []
#                 for ii in range(x.shape[0]):
#                     per = 99
#                     vector = x[ii].flatten().detach().cpu().numpy()
#                     vector = vector[vector > 0]
#                     threshold_pos = np.percentile(vector, per)
#                     vector = x[ii].flatten().detach().cpu().numpy()
#                     vector = vector[vector < 0]
#                     threshold_neg = np.percentile(vector, 100 - per)
#                     indices_gt = torch.nonzero(x[ii] > threshold_pos)
#                     indices_lt = torch.nonzero(x[ii] < threshold_neg)
#                     indices = torch.unique(torch.cat((indices_gt, indices_lt), dim=0))
#                     indices_list.append(indices)
#                 count_dict = Counter()
#                 for item in indices_list:
#                     count_dict.update(item.tolist())
#                 best_top_number = x.shape[0] // 20
#                 print('best_top_number', best_top_number)
#                 top_ = count_dict.most_common(best_top_number)
#                 top_values = [value for value, count in top_]
#                 top_values_tensor = torch.tensor(top_values).cuda()
#                 self.unique_top_indices = top_values_tensor
#                 tmp = []
#                 for i in range(x.shape[1]):
#                     if i not in self.unique_top_indices:
#                         tmp.append(i)
#                 self.unique_nottop_indices = torch.tensor(tmp).cuda()
#                 x_1 = x[:, self.unique_top_indices]
#                 x_2 = x[:, self.unique_nottop_indices]
#                 delta, zero_point = self.init_quantization_scale(x_2, self.channel_wise)
#                 self.delta_2, self.zero_point_2 = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()
#                 delta, zero_point = self.init_quantization_scale(x_1, self.channel_wise)
#                 self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()

#             self.inited.fill_(1)

#         # ===== 混合精度的前向传播逻辑 =====
#         if self.is_mixed_precision:
#             # 1. 创建掩码来区分正常值和异常值
#             inlier_mask = (x >= self.range_min) & (x <= self.range_max)
#             outlier_mask = ~inlier_mask

#             # 2. 创建一个空的输出张量
#             x_dequant = torch.zeros_like(x)

#             # 3. 处理正常值 (Inliers)
#             x_int_inlier = torch.round(x[inlier_mask] / self.delta) + self.zero_point
#             x_quant_inlier = torch.clamp(x_int_inlier, 0, self.n_levels - 1)
#             x_dequant[inlier_mask] = (x_quant_inlier - self.zero_point) * self.delta

#             # 4. 处理异常值 (Outliers)
#             x_int_outlier = torch.round(x[outlier_mask] / self.delta_outlier) + self.zero_point_outlier
#             x_quant_outlier = torch.clamp(x_int_outlier, 0, self.n_levels_outlier - 1)
#             x_dequant[outlier_mask] = (x_quant_outlier - self.zero_point_outlier) * self.delta_outlier

#             return x_dequant
#         # ================================

#         # ===== 以下是您原始代码的逻辑，保持不变 =====
#         if not self.two_part:
#             # start quantization
#             if index is None:
#                 if round_way == 'round':
#                     x_int = torch.round(x / self.delta) + self.zero_point
#                     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#                     x_dequant = (x_quant - self.zero_point) * self.delta
#                 elif round_way == 'floor':
#                     x_int = torch.floor(x / self.delta) + self.zero_point
#                     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#                     x_dequant = (x_quant - self.zero_point) * self.delta
#                 elif round_way == 'ceil':
#                     x_int = torch.ceil(x / self.delta) + self.zero_point
#                     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#                     x_dequant = (x_quant - self.zero_point) * self.delta
#                 else:
#                     raise Exception
#             else:
#                 if round_way == 'round':
#                     x_int = torch.round(x / self.delta[index]) + self.zero_point[index]
#                     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#                     x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
#                 elif round_way == 'floor':
#                     x_int = torch.floor(x / self.delta[index]) + self.zero_point[index]
#                     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#                     x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
#                 elif round_way == 'ceil':
#                     x_int = torch.ceil(x / self.delta[index]) + self.zero_point[index]
#                     x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#                     x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
#                 else:
#                     raise Exception
#         else:
#             if index is None:
#                 x_1 = x[:, self.unique_top_indices]
#                 x_2 = x[:, self.unique_nottop_indices]
#                 if self.split_quantization:
#                     self.negative_mask = torch.sign(x_1)
#                 if round_way == 'round':
#                     if not self.split_quantization:
#                         x_int_1 = torch.round(x_1 / self.delta) + self.zero_point
#                         x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
#                         x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
#                     else:
#                         x_1 = self.negative_mask * x_1
#                         x_int_1 = torch.round(x_1 / self.delta) + self.zero_point
#                         x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
#                         x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
#                         x_dequant_1 = self.negative_mask * x_dequant_1
#                     x_int_2 = torch.round(x_2 / self.delta_2) + self.zero_point_2
#                     x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
#                     x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
#                 elif round_way == 'floor':
#                     if not self.split_quantization:
#                         x_int_1 = torch.floor(x_1 / self.delta) + self.zero_point
#                         x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
#                         x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
#                     else:
#                         x_1 = self.negative_mask * x_1
#                         x_int_1 = torch.floor(x_1 / self.delta) + self.zero_point
#                         x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
#                         x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
#                         x_dequant_1 = self.negative_mask * x_dequant_1
#                     x_int_2 = torch.floor(x_2 / self.delta_2) + self.zero_point_2
#                     x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
#                     x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
#                 elif round_way == 'ceil':
#                     if not self.split_quantization:
#                         x_int_1 = torch.ceil(x_1 / self.delta) + self.zero_point
#                         x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
#                         x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
#                     else:
#                         x_1 = self.negative_mask * x_1
#                         x_int_1 = torch.ceil(x_1 / self.delta) + self.zero_point
#                         x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
#                         x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
#                         x_dequant_1 = self.negative_mask * x_dequant_1
#                     x_int_2 = torch.ceil(x_2 / self.delta_2) + self.zero_point_2
#                     x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
#                     x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
#                 else:
#                     raise Exception
#                 x_dequant = torch.zeros(x.shape).cuda()
#                 x_dequant[:, self.unique_top_indices] = x_dequant_1
#                 x_dequant[:, self.unique_nottop_indices] = x_dequant_2
#             else:  # index is not None logic
#                 # ... (this part of your original code remains unchanged)
#                 pass
#         return x_dequant

#     # The rest of the class methods (init_quantization_scale, quantize) remain unchanged
#     # as they are not used by the new mixed-precision mode.
#     # ... (Please paste your original init_quantization_scale and quantize methods here)
#     def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
#         delta, zero_point = None, None
#         if channel_wise:
#             x_clone = x.clone().detach()
#             if self.is_act:
#                 # n_channels = x_clone.shape[-1] if len(x.shape) == 3
#                 if len(x.shape) == 3:  # torch.Size([64, 197, 384])
#                     n_channels = x_clone.shape[-1]
#                 elif len(x.shape) == 4:  # torch.Size([64, 3, 224, 224]) or q k v to-do:
#                     n_channels = x_clone.shape[1]  # quantization dim=3
#                 elif len(x.shape) == 2:  # torch.Size([64, 384])
#                     n_channels = x_clone.shape[1]  # quantization dim=384
#                 else:
#                     raise NotImplementedError

#                 if len(x.shape) == 4:
#                     # x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
#                     x_max = x_clone.abs().max(dim=0)[0].max(dim=-1)[0].max(dim=-1)[0]
#                 elif len(x.shape) == 2:  #
#                     x_max = x_clone.abs().max(dim=0)[0]
#                 elif len(x.shape) == 3:
#                     x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
#                 else:
#                     raise NotImplementedError

#                 delta = x_max.clone()
#                 zero_point = x_max.clone()
#                 # determine the scale and zero point channel-by-channel
#                 for c in range(n_channels):
#                     if len(x.shape) == 3:
#                         delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, :, c], channel_wise=False)
#                     elif len(x.shape) == 4:
#                         delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, c, ...], channel_wise=False)
#                     else:
#                         delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, c], channel_wise=False)
#                 if len(x.shape) == 4:

#                     delta = delta.reshape(1, -1, 1, 1)
#                     zero_point = zero_point.reshape(1, -1, 1, 1)
#                 elif len(x.shape) == 2:
#                     delta = delta.reshape(1, -1)
#                     zero_point = zero_point.reshape(1, -1)
#                 elif len(x.shape) == 3:
#                     delta = delta.reshape(1, 1, -1)
#                     zero_point = zero_point.reshape(1, 1, -1)
#                 else:
#                     raise NotImplementedError
#             else:
#                 n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
#                 if len(x.shape) == 4:
#                     x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
#                 elif len(x.shape) == 2:
#                     x_max = x_clone.abs().max(dim=-1)[0]
#                 elif len(x.shape) == 3:
#                     x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
#                 else:
#                     raise NotImplementedError

#                 delta = x_max.clone()
#                 zero_point = x_max.clone()
#                 # determine the scale and zero point channel-by-channel
#                 for c in range(n_channels):
#                     if len(x.shape) == 3:
#                         delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:, :, c],
#                                                                                channel_wise=False)
#                     else:
#                         delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c],
#                                                                                channel_wise=False)
#                 if len(x.shape) == 4:
#                     delta = delta.reshape(-1, 1, 1, 1)
#                     zero_point = zero_point.reshape(-1, 1, 1, 1)
#                 elif len(x.shape) == 2:
#                     delta = delta.reshape(-1, 1)
#                     zero_point = zero_point.reshape(-1, 1)
#                 elif len(x.shape) == 3:
#                     delta = delta.reshape(1, 1, -1)
#                     zero_point = zero_point.reshape(1, 1, -1)
#                 else:
#                     raise NotImplementedError
#         else:
#             x_clone = x.clone().detach()
#             x_max = x_clone.max()
#             x_min = x_clone.min()
#             best_score = 1e+10
#             if self.is_act:
#                 search_range = [1]  # 0.999, 0.9999, 0.99999
#             else:
#                 search_range = [1]  # 0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1

#             if not self.split_quantization:
#                 for pct in search_range:
#                     try:
#                         new_max = torch.quantile(x_clone.reshape(-1), pct)
#                         new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
#                     except:
#                         new_max = torch.tensor(np.percentile(
#                             x_clone.reshape(-1).cpu(), pct * 100),
#                             device=x_clone.device,
#                             dtype=torch.float32)
#                         new_min = torch.tensor(np.percentile(
#                             x_clone.reshape(-1).cpu(), (1 - pct) * 100),
#                             device=x_clone.device,
#                             dtype=torch.float32)
#                     x_q, new_max, new_min = self.quantize(x_clone, new_max, new_min)
                    
#                     score = lp_loss(x_clone, x_q, p=2, reduction='all')
#                     if score < best_score:
#                         best_score = score
#                         delta = (new_max - new_min) / (2 ** self.n_bits - 1)
#                         zero_point = (- new_min / delta).round()
#             else:
#                 x_clone = torch.abs(x_clone)
#                 for pct in search_range:
#                     try:
#                         new_max = torch.quantile(x_clone.reshape(-1), pct)
#                         new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
#                     except:
#                         new_max = torch.tensor(np.percentile(
#                             x_clone.reshape(-1).cpu(), pct * 100),
#                             device=x_clone.device,
#                             dtype=torch.float32)
#                         new_min = torch.tensor(np.percentile(
#                             x_clone.reshape(-1).cpu(), (1 - pct) * 100),
#                             device=x_clone.device,
#                             dtype=torch.float32)
#                     x_q, new_max, new_min = self.quantize(x_clone, new_max, new_min)
#                     score = lp_loss(x_clone, x_q, p=2, reduction='all')
#                     if score < best_score:
#                         best_score = score
#                         delta = (new_max - new_min) / (2 ** (self.n_bits - 1) - 1)
#                         zero_point = (- new_min / delta).round()

#         return delta, zero_point

#     def quantize(self, x, max, min):
#         if not self.split_quantization:
#             if self.is_act:
#                 maxq = 2 ** self.n_bits - 1
#                 range_max, range_min = hod.find_outlier(7, x, 32)
#                 scale_real = (max - min) / maxq
#                 zero_real = (- min / scale_real).round()
#                 scale_range = (range_max - range_min) / maxq
#                 zero_range = (-range_min / scale_range).round()
#                 x_shape = x.shape
#                 x = x.flatten()
#                 x_temp = x.clone()
#                 xmin = x_temp - range_min
#                 xmax = range_max - x_temp
#                 tmp_in_range = (xmin >= 0) & (xmax >= 0)
#                 tmp_out_range = (xmin < 0) | (xmax < 0)
#                 in_operation_q = torch.clamp(torch.round(x / scale_range) + zero_range, 0, maxq)
#                 out_operation_q = torch.clamp(torch.round(x / scale_real) + zero_real, 0, maxq)
#                 x_temp[tmp_in_range] = in_operation_q[tmp_in_range]
#                 x_temp[tmp_out_range] = out_operation_q[tmp_out_range]
#                 in_operation_iq = scale_range * (x_temp - zero_range)
#                 out_operation_iq = scale_real * (x_temp - zero_real)
#                 x_temp[tmp_in_range] = in_operation_iq[tmp_in_range]
#                 x_temp[tmp_out_range] = out_operation_iq[tmp_out_range]
#                 x_temp = x_temp.reshape(x_shape)
#                 return x_temp, range_max, range_min
#             else:
#                 delta = (max - min) / (2 ** self.n_bits - 1)
#                 zero_point = (- min / delta).round()
#                 x_int = torch.round(x / delta)
#                 x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
#                 x_float_q = (x_quant - zero_point) * delta
#                 return x_float_q, max, min
#         else:
#             range_max, range_min = hod.find_outlier(7, x, 32)
#             delta = (max - min) / (2 ** self.n_bits - 1)
#             zero_point = (- min / delta).round()
#             x_int = torch.round(x / delta)
#             x_quant = torch.clamp(x_int + zero_point, 0, (self.n_levels // 2) - 1)
#             x_float_q = (x_quant - zero_point) * delta
#             return x_float_q
        
# class LogSqrt2Quantizer(nn.Module):
#     """
#     PyTorch Function that can be used for asymmetric quantization (also called uniform affine
#     quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
#     through' on the backward pass, ignoring the quantization that occurred.
#     Based on https://arxiv.org/abs/1806.08342.
#     :param n_bits: number of bit for quantization
#     :param channel_wise: if True, compute scale and zero_point in each channel
#     """
#     def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act:bool = False, bins:int = 128):
#         super(LogSqrt2Quantizer, self).__init__()
#         assert 2 <= n_bits <= 8, 'bitwidth not supported'
#         self.n_bits = n_bits
#         self.n_levels = 2 ** self.n_bits
#         self.delta = None
#         self.inited = False
#         self.channel_wise = channel_wise
#         self.bins = bins


#     def forward(self, x: torch.Tensor):

#         if self.inited is False:
#             self.delta = self.init_quantization_scale(x)
#             self.inited = True

#         # start quantization
#         x_dequant = self.quantize(x, self.delta)
#         return x_dequant

#     def init_quantization_scale(self, x: torch.Tensor):
#         delta = None
#         x_clone = x.clone().detach()
#         delta = x_clone.max()
#         best_score = 1e+10
#         for pct in [0.999, 0.9999, 0.99999]: #
#             try:
#                 new_delta = torch.quantile(x_clone.reshape(-1), pct)
#             except:
#                 new_delta = torch.tensor(np.percentile(
#                     x_clone.reshape(-1).cpu(), pct * 100),
#                     device=x_clone.device,
#                     dtype=torch.float32)
#             x_q = self.quantize(x_clone, new_delta)
#             score = lp_loss(x_clone, x_q, p=2, reduction='all')
#             if score < best_score:
#                 best_score = score
#                 delta = new_delta

#         return delta

#     def quantize(self, x, delta):      
#         from math import sqrt
#         x_int = torch.round(-1 * (x/delta).log2() * 2)
#         mask = x_int >= self.n_levels
#         x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#         odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
#         x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
#         x_float_q[mask] = 0

#         # x_int = torch.round(-1 * (x / delta).log2())
#         # mask = x_int >= self.n_levels
#         # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
#         # x_float_q = 2 ** (-1 * x_quant) * delta
#         # x_float_q[mask] = 0
#         #
#         return x_float_q




import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from collections import Counter
import numpy as np
import quant.horizonal_output_distribution_left_right as hod

def robust_kl_divergence(p, q):
    """最稳健的 KL 散度计算"""
    # 转为概率分布（使用 softmax）
    p_softmax = torch.nn.functional.softmax(p.flatten(), dim=0)
    q_softmax = torch.nn.functional.softmax(q.flatten(), dim=0)
    
    # 添加极小值
    p_softmax = p_softmax + 1e-10
    q_softmax = q_softmax + 1e-10
    
    # 归一化
    p_softmax = p_softmax / p_softmax.sum()
    q_softmax = q_softmax / q_softmax.sum()
    
    # 计算 KL
    kl = (p_softmax * (p_softmax.log() - q_softmax.log())).sum()
    
    return kl.item()

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act:bool = False,
                 two_part:bool = False, split_quantization: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.register_buffer('inited', torch.zeros(1))
        self.channel_wise = channel_wise
        self.is_act = is_act
        self.store_input = None
        self.two_part = two_part
        self.split_quantization = split_quantization
        self.negative_mask = None

    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def forward(self, x: torch.Tensor, round_way='round', index=None):

        if self.inited == 0:
            if not self.two_part:
                delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()
            else:

                indices_list = []
                for ii in range(x.shape[0]):

                    per = 99
                    vector = x[ii].flatten().detach().cpu().numpy()
                    vector = vector[vector > 0]
                    threshold_pos = np.percentile(vector, per)

                    vector = x[ii].flatten().detach().cpu().numpy()
                    vector = vector[vector < 0]
                    threshold_neg = np.percentile(vector, 100-per)

                    indices_gt = torch.nonzero(x[ii] > threshold_pos)
                    indices_lt = torch.nonzero(x[ii] < threshold_neg)
                    indices = torch.unique(torch.cat((indices_gt, indices_lt), dim=0))
                    indices_list.append(indices)

                count_dict = Counter()
                for item in indices_list:
                    count_dict.update(item.tolist())

                best_top_number = x.shape[0] // 20
                print('best_top_number', best_top_number)
                # _, top_values_tensor = torch.topk(outliers_count, best_top_number)
                top_ = count_dict.most_common(best_top_number)
                top_values = [value for value, count in top_]
                top_values_tensor = torch.tensor(top_values).cuda()
                self.unique_top_indices = top_values_tensor
                tmp = []
                for i in range(x.shape[1]):
                    if i not in self.unique_top_indices:
                        tmp.append(i)
                self.unique_nottop_indices = torch.tensor(tmp).cuda()
                x_1 = x[:, self.unique_top_indices]
                x_2 = x[:, self.unique_nottop_indices]

                delta, zero_point = self.init_quantization_scale(x_2, self.channel_wise)
                self.delta_2, self.zero_point_2 = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()

                delta, zero_point = self.init_quantization_scale(x_1, self.channel_wise)
                self.delta, self.zero_point = Parameter(delta).contiguous(), Parameter(zero_point).contiguous()


            self.inited.fill_(1)
        self.two_part = False
        if not self.two_part:
            # start quantization
            if index is None:

                if self.is_act:
                    x_dequant = self.init_quantization_scale_log(x,True)
                    kl_value = robust_kl_divergence(x_dequant, x)
                    print()
                    # score1 = lp_loss(x, x_dequant1, p=2, reduction='all')
                else:
                    x_int = torch.round(x / self.delta) + self.zero_point
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point) * self.delta
                    # score = lp_loss(x, x_dequant, p=2, reduction='all')
                    # print()

            else:
                if round_way == 'round':
                    x_int = torch.round(x / self.delta[index]) + self.zero_point[index]
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
                elif round_way == 'floor':
                    x_int = torch.floor(x / self.delta[index]) + self.zero_point[index]
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
                elif round_way == 'ceil':
                    x_int = torch.ceil(x / self.delta[index]) + self.zero_point[index]
                    x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                    x_dequant = (x_quant - self.zero_point[index]) * self.delta[index]
                else:
                    raise Exception
        else:
            if index is None:
                # start quantization
                x_1 = x[:, self.unique_top_indices]
                x_2 = x[:, self.unique_nottop_indices]

                if self.split_quantization:
                    self.negative_mask = torch.sign(x_1)

                if round_way == 'round':
                    if not self.split_quantization:

                        x_int_1 = torch.round(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                    else:

                        x_1 = self.negative_mask * x_1
                        x_int_1 = torch.round(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                        x_dequant_1 = self.negative_mask * x_dequant_1

                    x_int_2 = torch.round(x_2 / self.delta_2) + self.zero_point_2
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
                elif round_way == 'floor':
                    if not self.split_quantization:
                        x_int_1 = torch.floor(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                    else:
                        x_1 = self.negative_mask * x_1
                        x_int_1 = torch.floor(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                        x_dequant_1 = self.negative_mask * x_dequant_1

                    x_int_2 = torch.floor(x_2 / self.delta_2) + self.zero_point_2
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
                elif round_way == 'ceil':
                    if not self.split_quantization:
                        x_int_1 = torch.ceil(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                    else:
                        x_1 = self.negative_mask * x_1
                        x_int_1 = torch.ceil(x_1 / self.delta) + self.zero_point
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point) * self.delta
                        x_dequant_1 = self.negative_mask * x_dequant_1


                    x_int_2 = torch.ceil(x_2 / self.delta_2) + self.zero_point_2
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2) * self.delta_2
                else:
                    raise Exception

                x_dequant = torch.zeros(x.shape).cuda()
                x_dequant[:, self.unique_top_indices] = x_dequant_1
                x_dequant[:, self.unique_nottop_indices] = x_dequant_2
            else:
                # start quantization
                x_1 = x[self.unique_top_indices]
                x_2 = x[self.unique_nottop_indices]

                if round_way == 'round':
                    if not self.split_quantization:
                        x_int_1 = torch.round(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                    else:

                        x_1 = self.negative_mask[index] * x_1
                        x_int_1 = torch.round(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, (self.n_levels//2) - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                        x_dequant_1 = self.negative_mask[index] * x_dequant_1

                    x_int_2 = torch.round(x_2 / self.delta_2[index]) + self.zero_point_2[index]
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2[index]) * self.delta_2[index]
                elif round_way == 'floor':
                    if not self.split_quantization:
                        x_int_1 = torch.floor(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                    else:
                        x_1 = self.negative_mask[index] * x_1
                        x_int_1 = torch.floor(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, (self.n_levels//2) - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                        x_dequant_1 = self.negative_mask[index] * x_dequant_1

                    x_int_2 = torch.floor(x_2 / self.delta_2[index]) + self.zero_point_2[index]
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2[index]) * self.delta_2[index]
                elif round_way == 'ceil':
                    if not self.split_quantization:
                        x_int_1 = torch.ceil(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, self.n_levels - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                    else:
                        x_1 = self.negative_mask[index] * x_1
                        x_int_1 = torch.ceil(x_1 / self.delta[index]) + self.zero_point[index]
                        x_quant_1 = torch.clamp(x_int_1, 0, (self.n_levels//2) - 1)
                        x_dequant_1 = (x_quant_1 - self.zero_point[index]) * self.delta[index]
                        x_dequant_1 = self.negative_mask[index] * x_dequant_1

                    x_int_2 = torch.ceil(x_2 / self.delta_2[index]) + self.zero_point_2[index]
                    x_quant_2 = torch.clamp(x_int_2, 0, self.n_levels - 1)
                    x_dequant_2 = (x_quant_2 - self.zero_point_2[index]) * self.delta_2[index]
                else:
                    raise Exception
                x_dequant = torch.zeros(x.shape).cuda()
                x_dequant[self.unique_top_indices] = x_dequant_1
                x_dequant[self.unique_nottop_indices] = x_dequant_2
        return x_dequant


    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            if self.is_act:
                # n_channels = x_clone.shape[-1] if len(x.shape) == 3
                if len(x.shape) == 3: # torch.Size([64, 197, 384])
                    n_channels = x_clone.shape[-1]
                elif len(x.shape) == 4: # torch.Size([64, 3, 224, 224]) or q k v to-do:
                    n_channels = x_clone.shape[1] # quantization dim=3
                elif len(x.shape) == 2: # torch.Size([64, 384])
                    n_channels = x_clone.shape[1] # quantization dim=384
                else:
                    raise NotImplementedError

                if len(x.shape) == 4: 
                    # x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:# 
                    x_max = x_clone.abs().max(dim=0)[0]
                elif len(x.shape) == 3: 
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                # determine the scale and zero point channel-by-channel
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                    elif len(x.shape) == 4:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c,...], channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,c], channel_wise=False)
                if len(x.shape) == 4:

                    delta = delta.reshape(1, -1, 1, 1)
                    zero_point = zero_point.reshape(1, -1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.reshape(1, -1)
                    zero_point = zero_point.reshape(1, -1)
                elif len(x.shape) == 3:
                    delta = delta.reshape(1, 1, -1)
                    zero_point = zero_point.reshape(1, 1, -1)
                else:
                    raise NotImplementedError
            else: 
                n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
                if len(x.shape) == 4:
                    x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                elif len(x.shape) == 2:
                    x_max = x_clone.abs().max(dim=-1)[0]
                elif len(x.shape) == 3:
                    x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                else:
                    raise NotImplementedError

                delta = x_max.clone()
                zero_point = x_max.clone()
                # determine the scale and zero point channel-by-channel
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c],
                                                                               channel_wise=False)
                    else:
                        delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c],
                                                                               channel_wise=False)
                if len(x.shape) == 4:
                    delta = delta.reshape(-1, 1, 1, 1)
                    zero_point = zero_point.reshape(-1, 1, 1, 1)
                elif len(x.shape) == 2:
                    delta = delta.reshape(-1, 1)
                    zero_point = zero_point.reshape(-1, 1)
                elif len(x.shape) == 3:
                    delta = delta.reshape(1, 1, -1)
                    zero_point = zero_point.reshape(1, 1, -1)
                else:
                    raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            if self.is_act:
                search_range = [0.999,0.9999,0.99999]#0.999,0.9999,0.99999
            else:
                search_range = [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1]#0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1

            if not self.split_quantization:
                for pct in search_range:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                x_clone = torch.abs(x_clone)
                for pct in search_range:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                    x_q = self.quantize(x_clone, new_max, new_min)
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** (self.n_bits-1) - 1)
                        zero_point = (- new_min / delta).round()


        return delta, zero_point
    

    def init_quantization_scale_log(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            if self.is_act:
                # n_channels = x_clone.shape[-1] if len(x.shape) == 3
                if len(x.shape) == 3: # torch.Size([64, 197, 384])
                    n_channels = x_clone.shape[-1]
                elif len(x.shape) == 4: # torch.Size([64, 3, 224, 224]) or q k v to-do:
                    n_channels = x_clone.shape[1] # quantization dim=3
                elif len(x.shape) == 2: # torch.Size([64, 384])
                    n_channels = x_clone.shape[1] # quantization dim=384
                else:
                    raise NotImplementedError



                x_dequant = x_clone.clone()
                # determine the scale and zero point channel-by-channel
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        x_dequant[:,:,c] = self.init_quantization_scale_log(x_clone[:,:,c], channel_wise=False)
                    elif len(x.shape) == 4:
                       x_dequant[:,c,...] = self.init_quantization_scale_log(x_clone[:,c,...], channel_wise=False)
                    else:
                        x_dequant[:,c]  = self.init_quantization_scale_log(x_clone[:,c], channel_wise=False)

            else: 
                n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]

                x_dequant = x_clone.clone()
                # determine the scale and zero point channel-by-channel
                for c in range(n_channels):
                    if len(x.shape) == 3:
                        x_dequant[:,:,c]= self.init_quantization_scale_log(x_clone[:,:,c],
                                                                               channel_wise=False)
                    else:
                        x_dequant[c] = self.init_quantization_scale_log(x_clone[c],channel_wise=False)
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            if self.is_act:
                search_range = [1]#
            else:
                search_range = [1]#

            if not self.split_quantization:
                for pct in search_range:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                    x_dequant = self.quantize_log(x_clone, new_max, new_min)

            else:
                x_clone = torch.abs(x_clone)
                for pct in search_range:
                    try:
                        new_max = torch.quantile(x_clone.reshape(-1), pct)
                        new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                    except:
                        new_max = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), pct * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                        new_min = torch.tensor(np.percentile(
                            x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                            device=x_clone.device,
                            dtype=torch.float32)
                    x_dequant = self.quantize_log(x_clone, new_max, new_min)


        return x_dequant


    def quantize(self, x, max, min):
        if not self.split_quantization:
            if self.is_act:
                delta = (max - min) / (2 ** self.n_bits - 1)
                zero_point = (- min / delta).round()
                # we assume weight quantization is always signed
                x_int = torch.round(x / delta)
                x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
                x_float_q = (x_quant - zero_point) * delta
                return x_float_q
            else:
                delta = (max - min) / (2 ** self.n_bits - 1)
                zero_point = (- min / delta).round()
                # we assume weight quantization is always signed
                x_int = torch.round(x / delta)
                x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
                x_float_q = (x_quant - zero_point) * delta
                return x_float_q
        else:
            delta = (max - min) / (2 ** self.n_bits - 1)
            zero_point = (- min / delta).round()
            # we assume weight quantization is always signed
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int + zero_point, 0, (self.n_levels//2) - 1)
            x_float_q = (x_quant - zero_point) * delta
            return x_float_q
        


    def quantize_log(self, x, max, min):
        if not self.split_quantization:
            if self.is_act:
                if not torch.isnan(x).any():
                    maxq = 2 ** self.n_bits - 1
                    range_max, range_min = hod.find_outlier(7, x, 32)
                    scale_real = (max - min) / maxq
                    zero_real = (- min / scale_real).round()
                    scale_range = (range_max - range_min) / maxq
                    zero_range = (-range_min / scale_range).round()
                    x_shape = x.shape
                    x = x.flatten()
                    x_temp = x.clone()
                    xmin = x_temp - range_min
                    xmax = range_max - x_temp
                    tmp_in_range = (xmin >= 0) & (xmax >= 0)
                    tmp_out_range = (xmin < 0) | (xmax < 0)
                    in_operation_q = torch.clamp(torch.round(x / scale_range) + zero_range, 0, maxq)
                    out_operation_q = torch.clamp(torch.round(x / scale_real) + zero_real, 0, maxq)
                    x_temp[tmp_in_range] = in_operation_q[tmp_in_range]
                    x_temp[tmp_out_range] = out_operation_q[tmp_out_range]
                    in_operation_iq = scale_range * (x_temp - zero_range)
                    out_operation_iq = scale_real * (x_temp - zero_real)
                    x_temp[tmp_in_range] = in_operation_iq[tmp_in_range]
                    x_temp[tmp_out_range] = out_operation_iq[tmp_out_range]
                    x_temp = x_temp.reshape(x_shape)
                    return x_temp

                else:
                    delta = (max * 0.999 - min *0.999) / (2 ** self.n_bits - 1)
                    zero_point = (- min / delta).round()
                    # we assume weight quantization is always signed
                    x_int = torch.round(x / delta)
                    x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
                    x_float_q = (x_quant - zero_point) * delta
                    return x_float_q
                
            else:

                delta = (max - min) / (2 ** self.n_bits - 1)
                zero_point = (- min / delta).round()
                # we assume weight quantization is always signed
                x_int = torch.round(x / delta)
                x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
                x_float_q = (x_quant - zero_point) * delta
                return x_float_q
        else:
            delta = (max - min) / (2 ** self.n_bits - 1)
            zero_point = (- min / delta).round()
            # we assume weight quantization is always signed
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int + zero_point, 0, (self.n_levels//2) - 1)
            x_float_q = (x_quant - zero_point) * delta
            return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, is_act:bool = False):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise


    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta = self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]: #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):      
        from math import sqrt
        x_int = torch.round(-1 * (x/delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
        x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
        x_float_q[mask] = 0
        return x_float_q
