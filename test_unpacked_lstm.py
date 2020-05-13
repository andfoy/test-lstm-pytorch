
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.utils.rnn as rnn

# torch.backends.cudnn.enabled = True  # False

lstm_gpu = nn.LSTM(10, 6, bidirectional=True, num_layers=2,
                   cat_layer_fwd_bwd_states=False, batch_first=True)
lstm_gpu_fw = nn.LSTM(10, 6, bidirectional=False, num_layers=2,
                      cat_layer_fwd_bwd_states=True, batch_first=True)
lstm_gpu_bw = nn.LSTM(10, 6, bidirectional=False, num_layers=2,
                      cat_layer_fwd_bwd_states=True, batch_first=True)
lstm_cpu = nn.LSTM(10, 6, bidirectional=True, num_layers=2,
                   cat_layer_fwd_bwd_states=False, batch_first=True)

lstm_cpu.load_state_dict(lstm_gpu.state_dict())
lstm_gpu.to('cuda')
lstm_gpu_fw.to('cuda')
lstm_gpu_bw.to('cuda')


state_dict = lstm_gpu.state_dict()
fwd_weights = {k: state_dict[k] for k in state_dict
               if not k.endswith('reverse')}
bwd_weights = {k.replace('_reverse', ''): state_dict[k] for k in state_dict
               if k.endswith('reverse')}
lstm_gpu_fw.load_state_dict(fwd_weights)
lstm_gpu_bw.load_state_dict(bwd_weights)


# lstm.to('cuda')
for w, n in zip(lstm_cpu._flat_weights, lstm_cpu._flat_weights_names):
    print(f'{n} sizes: {w.size()}')

for w, n in zip(lstm_gpu._flat_weights, lstm_gpu._flat_weights_names):
    print(f'{n} sizes: {w.size()}')


# print(lstm)
input_size = 10
hidden_size = 6
num_layers = 2
seq_length = 7
batch = 6

hx_val = torch.randn(num_layers * 2, batch, hidden_size)
cx_val = torch.randn(num_layers * 2, batch, hidden_size)

x = torch.randn(batch, seq_length, input_size)  # .to('cuda')
rev_x = []
for input in x.split(1, 0):
    split_input = input.split(1, 1)
    reversed_input = split_input[::-1]
    reversed_input = torch.cat(reversed_input, 1)
    rev_x.append(reversed_input)

rev_x = torch.cat(rev_x, 0) 

with torch.no_grad():
    cpu_out, _ = lstm_cpu(x, (hx_val, cx_val))

with torch.no_grad():
    gpu_out, _ = lstm_gpu(x.to('cuda'), (hx_val.to('cuda'), cx_val.to('cuda')))


hx = hx_val.split(1, 0)
hx_fwd = torch.cat([h for i, h in enumerate(hx) if i % 2 == 0], 0)
hx_bwd = torch.cat([h for i, h in enumerate(hx) if i % 2 != 0], 0)

cx = cx_val.split(1, 0)
cx_fwd = torch.cat([h for i, h in enumerate(cx) if i % 2 == 0], 0)
cx_bwd = torch.cat([h for i, h in enumerate(cx) if i % 2 != 0], 0)

with torch.no_grad():
    gpu_fwd_out, _ = lstm_gpu_fw(x.to('cuda'), (hx_fwd.to('cuda'), cx_fwd.to('cuda')))

with torch.no_grad():
    gpu_bwd_out, _ = lstm_gpu_bw(rev_x.to('cuda'), (hx_bwd.to('cuda'), cx_bwd.to('cuda')))


print(cpu_out - gpu_out.to('cpu'))

