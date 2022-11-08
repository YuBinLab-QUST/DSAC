import math
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
            Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self, seqlen,hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        self.attention_size = int(hidden_size)
        self.seqlen=seqlen

        self.query = nn.Linear(hidden_size, self.attention_size)
        self.key = nn.Linear(hidden_size, self.attention_size)
        self.value = nn.Linear(hidden_size, self.attention_size)


        self.attn_dropout = nn.Dropout(hidden_dropout_prob)
        self.BatchNorm=nn.BatchNorm1d(seqlen)
        self.relu=nn.ReLU(inplace=True)

        """
            After doing self-attention,make a feedforward fully connected LayerNorm output
        """
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (1, self.attention_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        a = nn.BatchNorm1d(self.seqlen)
        input = a(input)
        input = self.relu(input)
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)
        #print(q.shape)

        Q = self.transpose_for_scores(q)
        K = self.transpose_for_scores(k)
        V = self.transpose_for_scores(v)

        """
            Take the dot product between "query" and "key" to get the raw attention scores.
        """
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        #print(attention_scores.shape)

        attention_scores = attention_scores / math.sqrt(self.attention_size)
        """
            Normalize the attention scores to probabilities.
        """
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        """
            This is actually dropping out entire tokens to attend to, which might
            seem a bit unusual, but is taken from the original Transformer paper.
        """
        attention_probs = self.attn_dropout(attention_probs)
        output = torch.matmul(attention_probs, V)
        output =output.permute(0, 2, 1, 3).contiguous()
        output_shape =output.size()[:-2] + (self.attention_size,)
        output =output.view(*output_shape)
        output = self.dense(output)
        output = self.out_dropout(output)
        #print(output.shape)

        result = self.LayerNorm(output + input)
        result=result.permute(0, 2, 1)
        return result
