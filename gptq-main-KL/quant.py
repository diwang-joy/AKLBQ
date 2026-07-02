import numpy as np
import torch
import torch.nn as nn
import horizonal_output_distribution_left_right as hod
import argparse
count_1,count_2,count_3,count_4,x_num,percentage = 0,0,0,0,0,0


def quantize(x, scale_real, zero_real, maxq, range_max, range_min):

    scale_range = (range_max - range_min) / maxq
   
    zero_range = torch.round(-range_min / scale_range)
    if maxq < 0:
        return (x > scale_real / 2).float() * scale_real + (x < zero_real / 2).float() * zero_real
    
    ################# orginal x improved version start ###########################
    x_temp = x.clone()
    scale_real_temp = scale_real.clone()
    zero_real_temp = zero_real.clone()
    range_max_temp = range_max.clone()
    range_min_temp = range_min.clone()
    scale_range_temp = scale_range.clone()
    zero_range_temp = zero_range.clone()


    xmin = x_temp - range_min_temp
    xmax = range_max_temp - x_temp
    tmp_in_range = (xmin >= 0) & (xmax >= 0)
    tmp_out_range = (xmin < 0) | (xmax < 0)
 
    masked_scale_range = scale_range_temp[tmp_in_range]
    masked_scale_real = scale_real_temp[tmp_out_range]
    masked_zero_range = zero_range_temp[tmp_in_range]
    masked_zero_real = zero_real_temp[tmp_out_range]
    x_temp[tmp_in_range] = torch.clamp(torch.round(x_temp[tmp_in_range] / masked_scale_range) + masked_zero_range, 0, maxq)
   
    x_temp[tmp_out_range] = torch.clamp(torch.round(x_temp[tmp_out_range] / masked_scale_real) + masked_zero_real, 0, 15)
    
    x_temp[tmp_in_range] = masked_scale_range * (x_temp[tmp_in_range] - masked_zero_range)
    x_temp[tmp_out_range] = masked_scale_real * (x_temp[tmp_out_range] - masked_zero_real)

    return x_temp

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        self.register_buffer('xmax_outlier', torch.zeros(shape))
        self.register_buffer('xmin_outlier', torch.zeros(shape))


    def configure(
        self,
        bins, left, right, bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ): 
    ########################################################################
        self.bins = bins
        self.left = left
        self.right = right
    ########################################################################
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, bins, left, right, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        #######################################################################33333
        self.bins = self.bins
        self.left = self.left
        self.right = self.right
        ##############################################################################
        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        self.xmin_outlier = torch.zeros(x.shape[0], device=dev)
        self.xmax_outlier = torch.zeros(x.shape[0], device=dev)
        real_xmin = torch.minimum(x.min(1)[0], tmp)
        real_xmax = torch.maximum(x.max(1)[0], tmp)

        for i in range(0,x.shape[0]):
            self.xmax_outlier[i],self.xmin_outlier[i] = hod.find_outlier(x[i],7)





        if self.sym:
            real_xmax = torch.maximum(torch.abs(real_xmin), real_xmax)
            tmp = real_xmin < 0
            if torch.any(tmp):
                real_xmin[tmp] = -real_xmax[tmp]
        tmp = (real_xmin == 0) & (real_xmax == 0)
        real_xmin[tmp] = -1
        real_xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = real_xmax
          self.zero = real_xmin
        else:
          self.scale = (real_xmax - real_xmin) / 15

        
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-real_xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * real_xmin
                xmax1 = p * real_xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            self.xmax_outlier = self.xmax_outlier.reshape(shape)
            self.xmin_outlier = self.xmin_outlier.reshape(shape)
            return count_1,count_2,count_3,count_4,x_num
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)
        
    
        

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq, self.xmax_outlier, self.xmin_outlier)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
except:
    print('CUDA extension not installed.')

# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
            else:
                x = x.float()
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

def make_quant3(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

















































# def quantize(x, scale_real, zero_real, maxq, range_max, range_min):

#     global cnt_Lbit
#     global cnt_Hbit

#     scale_range = (range_max - range_min) / maxq
   
#     zero_range = torch.round(-range_min / scale_range)
#     if maxq < 0:
#         return (x > scale_real / 2).float() * scale_real + (x < zero_real / 2).float() * zero_real
    
#     ################# orginal x improved version start ###########################
#     x_temp = x.clone()
#     scale_real_temp = scale_real.clone()
#     zero_real_temp = zero_real.clone()
#     range_max_temp = range_max.clone()
#     range_min_temp = range_min.clone()
#     scale_range_temp = scale_range.clone()
#     zero_range_temp = zero_range.clone()

#     xmin = x_temp - range_min_temp
#     xmax = range_max_temp - x_temp
#     tmp_in_range = (xmin >= 0) & (xmax >= 0)
#     tmp_out_range = (xmin < 0) | (xmax < 0)
 
#     masked_scale_range = scale_range_temp[tmp_in_range]
#     masked_scale_real = scale_real_temp[tmp_out_range]
#     masked_zero_range = zero_range_temp[tmp_in_range]
#     masked_zero_real = zero_real_temp[tmp_out_range]

#     #abc = zero_real_temp[masked_zero_real]
#     cnt_Hbit += zero_real_temp[tmp_out_range].size()[0]
#     with open('cnt_Hbit.txt', 'a') as file:
#          file.write(str(cnt_Hbit) + '\n')
#     cnt_Lbit += zero_range_temp[tmp_in_range].size()[0]
#     with open('cnt_Lbit.txt', 'a') as file:
#          file.write(str(cnt_Lbit) + '\n')
    


#     x_temp[tmp_in_range] = torch.clamp(torch.round(x_temp[tmp_in_range] / masked_scale_range) + masked_zero_range, 0, maxq)
   
#     x_temp[tmp_out_range] = torch.clamp(torch.round(x_temp[tmp_out_range] / masked_scale_real) + masked_zero_real, 0, maxq)
    
#     x_temp[tmp_in_range] = masked_scale_range * (x_temp[tmp_in_range] - masked_zero_range)
#     x_temp[tmp_out_range] = masked_scale_real * (x_temp[tmp_out_range] - masked_zero_real)
#     ################# orginal x improved version  end ###########################
    
#     # q = torch.clamp(torch.round(x / scale_real) + zero_real, 0, maxq)
#     # return scale_real * (q - zero_real)
#     return x_temp
