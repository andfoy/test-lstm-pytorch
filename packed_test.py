
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.utils.rnn as rnn

# torch.backends.cudnn.enabled = True  # False

lstm_gpu = nn.RNN(10, 6, bidirectional=True, num_layers=2,
                   cat_layer_fwd_bwd_states=False, batch_first=True)
lstm_gpu_fw = nn.RNN(10, 6, bidirectional=False, num_layers=2,
                      cat_layer_fwd_bwd_states=True, batch_first=True)
lstm_gpu_bw = nn.RNN(10, 6, bidirectional=False, num_layers=2,
                      cat_layer_fwd_bwd_states=True, batch_first=True)
lstm_cpu = nn.RNN(10, 6, bidirectional=True, num_layers=2,
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

x = torch.randn(batch, seq_length, input_size)  # .to('cuda')
lengths = [7, 5, 5, 2, 1, 1]
rev_x = []
for input, length in zip(x.split(1, 0), lengths):
    split_input = input.split(1, 1)
    reversed_input = split_input[:length][::-1] + split_input[length:]
    reversed_input = torch.cat(reversed_input, 1)
    rev_x.append(reversed_input)

rev_x = torch.cat(rev_x, 0) 

packed_x = rnn.pack_padded_sequence(x, lengths, batch_first=True)
rev_packed_x = rnn.pack_padded_sequence(rev_x, lengths, batch_first=True)

# x = torch.rand(10, 1, 256) #.to('cuda')
# y = torch.rand(batch, seq_length, hidden_size * 2) # .to('cuda')
# y = rnn.pack_padded_sequence(y, lengths, batch_first=True).data

with torch.no_grad():
    cpu_out, _ = lstm_cpu(packed_x, hx_val)


# lstm.to('cuda')
# x = x.to('cuda')

with torch.no_grad():
    gpu_out, _ = lstm_gpu(packed_x.to('cuda'), hx_val.to('cuda'))

# hx_fwd, hx_bwd = hx_val.split(2, 0)
hx = hx_val.split(1, 0)
hx_fwd = torch.cat([h for i, h in enumerate(hx) if i % 2 == 0], 0)
hx_bwd = torch.cat([h for i, h in enumerate(hx) if i % 2 != 0], 0)

with torch.no_grad():
    gpu_fwd_out, _ = lstm_gpu_fw(packed_x.to('cuda'), hx_fwd.to('cuda'))

with torch.no_grad():
    gpu_bwd_out, _ = lstm_gpu_bw(rev_packed_x.to('cuda'), hx_bwd.to('cuda'))

bwd_unpacked, _ = rnn.pad_packed_sequence(gpu_bwd_out, batch_first=True)
rev_bwd = []
for input, length in zip(bwd_unpacked.split(1, 0), lengths):
    split_input = input.split(1, 1)
    reversed_input = split_input[:length][::-1] + split_input[length:]
    reversed_input = torch.cat(reversed_input, 1)
    rev_bwd.append(reversed_input)

rev_bwd = torch.cat(rev_bwd, 0)
rev_bwd = rnn.pack_padded_sequence(rev_bwd, lengths, batch_first=True)

print(cpu_out.data - gpu_out.data.to('cpu'))

