# This is a simple frame of Transformer
# Import Package
import torch
import torch.nn as nn
import torch.optim as optim

# Transformer Class
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers, num_heads, dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask)
        output = self.fc(dec_output)
        return output

# Encoder and Decoder Class
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, mask):
        output = tgt
        for layer in self.layers:
            output = layer(output, enc_output, mask)
        return output

# Layer Class
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(hidden_dim, dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask):
        output = self.self_attn(src, src, src, mask)
        output = self.dropout(output)
        output = output + src
        output = self.layer_norms[0](output)
        output = self.feed_forward(output)
        output = self.dropout(output)
        output = output + src
        output = self.layer_norms[1](output)
        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(hidden_dim, dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, mask):
        output = self.self_attn(tgt, tgt, tgt, mask)
        output = self.dropout(output)
        output = output + tgt
        output = self.layer_norms[0](output)
        output = self.enc_dec_attn(output, enc_output, enc_output, mask)
        output = self.dropout(output)
        output = output + tgt
        output = self.layer_norms[1](output)
        output = self.feed_forward(output)
        output = self.dropout(output)
        output = output + tgt
        output = self.layer_norms[2](output)
        return output

# Multi-head Attention and Position-wise Feed Forward
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, value = -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.fc_o(output)
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Train example
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch_input, batch_target in data_loader:
        batch_input = batch_input.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad()
        output = model(batch_input, batch_target)
        loss = criterion(output.view(-1, output.shape[-1]), batch_target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

# Example usage
# model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# for epoch in range(num_epochs):
#     train_loss = train(model, data_loader, optimizer, criterion, device)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}")
