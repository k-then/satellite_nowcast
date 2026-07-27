import torch


def float_to_16bitint(tensor, frac_bits=13):
    scale = 2.0 ** frac_bits
    clamped = torch.clamp(torch.round(tensor * scale), min=-32768, max=32767)
    return clamped.to(torch.int16)


def load_lut(filename):
    with open(filename, "r") as f:
        hex_lines = f.read().splitlines()
    
    int_vals = []
    for line in hex_lines:
        line = line.strip()
        if not line:
            continue
        val = int(line, 16)
        
        if val >= 0x8000:
            val -= 0x10000
            
        int_vals.append(val)
        
    return torch.tensor(int_vals, dtype=torch.int16)


def apply_lut(tensor, lut_table, frac_bits=13):
    x_float = tensor.float() / (2.0 ** frac_bits)
    x_clamped = torch.clamp(x_float, min=-8.0, max=8.0)
    index = torch.clamp(torch.floor((x_clamped + 8.0) * (256.0 / 16.0)).long(), 0, 255)
    print("DEBUG Indices:", index)
    return lut_table[index]


if __name__ == "__main__":
    sigmoid_table = load_lut("sigmoid_lut_q015.mem")
    tanh_table = load_lut("tanh_lut_q015.mem")

    print("First 3 elements of LUT:", sigmoid_table[:3])

    test_floats = torch.tensor([-10.0, -8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 10.0])

    q_inputs = float_to_16bitint(test_floats, frac_bits=12)

    sig_outputs_q15 = apply_lut(q_inputs, sigmoid_table, frac_bits=12)
    tanh_outputs_q15 = apply_lut(q_inputs, tanh_table, frac_bits=12)

    sig_outputs_float = sig_outputs_q15.float() / (2.0 ** 15)
    tanh_outputs_float = tanh_outputs_q15.float() / (2.0 ** 15)

    print("Inputs (Float):", test_floats)
    print("Sigmoid (LUT): ", sig_outputs_float)
    print("Tanh (LUT):    ", tanh_outputs_float)