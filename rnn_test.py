import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn

#torch.backends.cudnn.enabled = True  # False

lstm = nn.LSTM(10, 6, bidirectional=True, num_layers=3,
               cat_layer_fwd_bwd_states=False, batch_first=True)


# lstm.to('cuda')
for w, n in zip(lstm._flat_weights, lstm._flat_weights_names):
    print(f'{n} sizes: {w.size()}')
#print(lstm._flat_weights_names)
#print(lstm._original_flat_names)
optimizer = optim.SGD(lstm.parameters(), lr=0.0001, momentum=0.9)
optimizer.zero_grad()

print(lstm)
input_size = 10
hidden_size = 6
num_layers = 2
seq_length = 7
batch = 6

x = torch.randn(batch, seq_length, input_size) # .to('cuda')
lengths = [7, 5, 5, 2, 1, 1]
x = rnn.pack_padded_sequence(x, lengths, batch_first=True)

# x = torch.rand(10, 1, 256) #.to('cuda')
# y = torch.rand(batch, seq_length, hidden_size * 2) # .to('cuda')
# y = rnn.pack_padded_sequence(y, lengths, batch_first=True).data

with torch.no_grad():
    cpu_out, _ = lstm(x)


lstm.to('cuda')
x = x.to('cuda')

with torch.no_grad():
   gpu_out, _ = lstm(x)

# print(cpu_out.data - gpu_out.data.to('cpu'))

#print("Forward")

#criterion = nn.MSELoss()
#loss = criterion(out.data, y)
#loss.backward()
