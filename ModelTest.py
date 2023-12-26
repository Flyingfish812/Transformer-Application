from TransformerFrame import *
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

class Batch:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
        self.src_mask = (src != 0).unsqueeze(-2)
        self.tgt_mask = (tgt != 0).unsqueeze(-2)

def data_generator(V, batch_size, num_batch):
    for _ in range(num_batch):
        # data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, 10)))
        data = torch.from_numpy(V*np.random.rand(8, batch_size))
        data[:, 0] = 1
        data = data.to(torch.float32)
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        yield Batch(source, target)

model = Transformer(input_dim=1, output_dim=1, hidden_dim=32, num_layers=2, num_heads=8, dropout=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# model_optimizer = get_std_opt(model)
# criterion = LabelSmoothing(size=11, padding_idx=0, smoothing=0.1)
criterion = nn.CrossEntropyLoss()
# loss = SimpleLossCompute(model.generator, criterion, model_optimizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        batch_input = batch.src.to(device)
        batch_target = batch.tgt.to(device)
        batch_input_mask = batch.src_mask.to(device)
        batch_output_mask = batch.tgt_mask.to(device)
        optimizer.zero_grad()
        output = model(batch_input, batch_target, batch_input_mask, batch_output_mask)
        loss = criterion(output.view(-1, output.shape[-1]), batch_target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

data_loader = data_generator(11, 32, 16)
if __name__ == "__main__":
    train(model, data_loader, optimizer, criterion, device)

# def run(model, loss, epochs = 10):
#     for epoch in range(epochs):
#         model.train()
#         run_epoch(data_generator(11, 20, 30), model, loss)
#         model.eval()
#         run_epoch(data_generator(11, 20, 5), model, loss)
    
#     model.eval()
#     source = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]))
#     source_mask = Variable(torch.ones(1, 1, 10))
#     result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
#     print(result)

# if __name__ == '__main__':
#     run(model, loss)