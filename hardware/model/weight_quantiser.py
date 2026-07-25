import torch
import torch.nn as nn
import numpy as np


# Checks the mapping of layer names
def check_layer_name(checkpoint):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    for key in state_dict.keys():
        print(key)


# Checks max size of data held in model state layers
def check_model_size(checkpoint):
    model_state = checkpoint['model_state']
    for name, tensor in model_state.items():
        if 'weight' in name or 'bias' in name:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            abs_max = max(abs(min_val), abs(max_val))
            
            print(f"Layer: {name}")
            print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}, Abs Max: {abs_max:.4f}")


def float_to_q015(tensor):
    clamped = torch.clamp(torch.round(tensor * 32768.0), min=-32768, max=32767)
    return clamped.to(torch.int16)


def torch_conv(state_dict):
    quantised_weight = {}
    for name, param in state_dict.items():
        if "weight" in name or "bias" in name:
            q_param = float_to_q015(param).numpy()
        quantised_weight[name] = q_param
        print(f"Quantized {name:35s} -> min: {q_param.min():6d}, max: {q_param.max():6d}")
    return quantised_weight


def int16_to_hex(val):
    return f"{(int(val) & 0xFFFF):04X}"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


def gen_lut(func, filename):
    x = np.linspace(-8.0,8.0,256)
    y_float = func(x)

    y_int = np.round(y_float * 32768.0)
    y_clamped = np.clip(y_int, -32768, 32767).astype(np.int16)

    with open(filename, "w") as f:
        for val in y_clamped:
            f.write(f"{int16_to_hex(val)}\n")


checkpoint_path = "checkpoints_v6/best.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint.get("model_state", checkpoint)
# check_layer_name(checkpoint)
# check_model_size(checkpoint)

np.savez_compressed("quantized_weights_q015.npz", **torch_conv(state_dict))
print("\nSaved quantized weights to quantized_weights_q015.npz!")

gen_lut(sigmoid, "sigmoid_lut_q015.mem")
gen_lut(tanh, "tanh_lut_q015.mem")
