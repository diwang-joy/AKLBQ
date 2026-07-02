import torch.nn as nn
import torch.nn.functional as F
from utils.build_model import MatMul
from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul
from copy import deepcopy


def quant_model(model, input_quant_params={}, weight_quant_params={}):
    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2['log_quant'] = True

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = True

    # # ==================== 建议的修改 ====================
    # # 在这里为激活值量化器启用混合精度模式
    # # 这会影响所有使用 input_quant_params_channel 的 QuantLinear 模块
    # print("Enabling Mixed-Precision Quantization for Activations...")
    # input_quant_params_channel['is_mixed_precision'] = True
    # # 您可以根据需要调整异常值的比特数，这里使用 4-bit 作为示例
    # input_quant_params_channel['n_bits_outlier'] = 4
    # # =====================================================

    module_dict = {}
    for name, m in model.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        if isinstance(m, nn.Conv2d):
            # Embedding Layer
            idx = idx + 1 if idx != 0 else idx
            new_m = QuantConv2d(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.groups,
                m.bias is not None,
                input_quant_params,
                weight_quant_params
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            # if 'qkv' in name or 'fc1' in name or 'reduction' in name:
            if 'qkv' in name or 'fc1' in name:
                weight_quant_params['two_part'] = True
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params)
                weight_quant_params.pop('two_part')
            else:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2)
            else:
                new_m = QuantMatMul(input_quant_params)
            setattr(father_module, name[idx:], new_m)

    return model


def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
        # if isinstance(m, (QuantLinear, QuantMatMul)):
        #     m.set_quant_state(input_quant, weight_quant)
        # if isinstance(m, QuantConv2d):
        #     m.set_quant_state(input_quant, True)
