import torch
import torch.nn as nn

lstm = nn.LSTM(256, 128, bidirectional=True, num_layers=3,
               cat_layer_fwd_bwd_states=True)

x = torch.rand(10, 1, 256)
out = lstm(x)
