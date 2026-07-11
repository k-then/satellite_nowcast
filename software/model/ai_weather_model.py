import torch
print(torch.__version__)
print(torch.cuda.is_available())      # should print True
print(torch.cuda.get_device_name(0))  # should print your RTX 4070 laptop GPU
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")