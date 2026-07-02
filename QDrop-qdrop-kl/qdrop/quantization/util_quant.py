import torch
import qdrop.quantization.horizonal_output_distribution_left_right as hod


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_per_channel_affine(self,x, scale, zero_point, ch_axis, quant_min, quant_max): #第一维度
    if self.bit == 4:
        # #
        # cur_max_range = torch.zeros(x.size(0))
        # cur_min_range = torch.zeros(x.size(0))
        # for i in range(x.size(0)):
        #     row = x[i]
        #     # th_r,th_l = hodlr.find_outlier(2**bits-1,row,bins)
        #     th_r,th_l = hod.find_outlier(15,row,65)
        #     cur_max_range[i] = float(th_r) 
        #     cur_min_range[i] = float(th_l)
            
        # cur_max_range = cur_max_range.to(torch.device("cuda:0"))
        # cur_min_range = cur_min_range.to(torch.device("cuda:0"))
        # # 
        # cur_min_range = cur_min_range.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # cur_max_range = cur_max_range.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # new_shape = [1] * len(x.shape)
        # new_shape[ch_axis] = x.shape[ch_axis]
        # scale = scale.reshape(new_shape)
        # zero_point = zero_point.reshape(new_shape)
        # scale_range = (cur_max_range - cur_min_range) / float(quant_max - quant_min)
        # # scale_range = torch.max(scale_range,self.eps)
        # zero_point_range = quant_min - torch.round(cur_min_range / scale_range)
        # zero_point_range = torch.clamp(zero_point_range,quant_min,quant_max)
        # #
        # x_temp = x.clone()
        # xmin = x_temp - cur_min_range
        # xmax = cur_max_range - x_temp
        # tmp_in_range = (xmin >= 0) & (xmax >= 0)
        # tmp_out_range = (xmin < 0) | (xmax < 0)
        # in_operation_q =  torch.clamp(torch.round(x / scale_range)  + zero_point_range, quant_min, quant_max)
        # out_operation_q = torch.clamp(torch.round(x / scale) + zero_point, quant_min, quant_max)
        # #
        # x_temp[tmp_in_range] = in_operation_q[tmp_in_range]
        # x_temp[tmp_out_range] = out_operation_q[tmp_out_range]
        # #
        # in_operation_iq = scale_range * (x_temp - zero_point_range)
        # out_operation_iq = scale * (x_temp - zero_point)
        # x_temp[tmp_in_range] = in_operation_iq[tmp_in_range]
        # x_temp[tmp_out_range] = out_operation_iq[tmp_out_range]
        # #
        # return x_temp
        new_shape = [1] * len(x.shape)
        new_shape[ch_axis] = x.shape[ch_axis]
        scale = scale.reshape(new_shape)
        zero_point = zero_point.reshape(new_shape)
        x_int = round_ste(x / scale) + zero_point
        x_quant = torch.clamp(x_int, quant_min, quant_max)
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant
    else:
        new_shape = [1] * len(x.shape)
        new_shape[ch_axis] = x.shape[ch_axis]
        scale = scale.reshape(new_shape)
        zero_point = zero_point.reshape(new_shape)
        x_int = round_ste(x / scale) + zero_point
        x_quant = torch.clamp(x_int, quant_min, quant_max)
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant

def fake_quantize_learnable_per_tensor_affine_training(self,x, scale, zero_point, quant_min, quant_max, grad_factor): # 全部为1个
    if self.bit == 4:
        # #
        # cur_max_range ,cur_min_range = hod.find_outlier(7,x,32)
        # cur_max_range = torch.tensor(cur_max_range).to('cuda')
        # cur_min_range = torch.tensor(cur_min_range).to('cuda')
        # # 
        # scale = grad_scale(scale, grad_factor)
        # scale_range = (cur_max_range - cur_min_range) / float(quant_max - quant_min)
        # # scale_range = torch.max(scale,self.eps)
        # scale_range = grad_scale(scale_range, grad_factor)
        # zero_point_range = quant_min - torch.round(cur_min_range / scale_range)
        # zero_point_range = torch.clamp(zero_point_range,quant_min,quant_max)
        # x_temp = x.clone()
        # xmin = x_temp - cur_min_range
        # xmax = cur_max_range - x_temp
        # tmp_in_range = (xmin >= 0) & (xmax >= 0)
        # tmp_out_range = (xmin < 0) | (xmax < 0)
        # in_operation_q =  torch.clamp(torch.round(x / scale_range)  + zero_point_range, quant_min, quant_max)
        # out_operation_q = torch.clamp(torch.round(x / scale) + zero_point, quant_min, quant_max)
        # #
        # x_temp[tmp_in_range] = in_operation_q[tmp_in_range]
        # x_temp[tmp_out_range] = out_operation_q[tmp_out_range]
        # #
        # in_operation_iq = scale_range * (x_temp - zero_point_range)
        # out_operation_iq = scale * (x_temp - zero_point)
        # x_temp[tmp_in_range] = in_operation_iq[tmp_in_range]
        # x_temp[tmp_out_range] = out_operation_iq[tmp_out_range]
        # return x_temp
        scale = grad_scale(scale, grad_factor)
        x_int = round_ste(x / scale) + zero_point
        x_quant = torch.clamp(x_int, quant_min, quant_max)
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant
    else:
        scale = grad_scale(scale, grad_factor)
        x_int = round_ste(x / scale) + zero_point
        x_quant = torch.clamp(x_int, quant_min, quant_max)
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant


def fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)
