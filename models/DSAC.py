import torch.nn as nn
import torch
import models.self_attention as Att

class dsac(nn.Module):

    def __init__(self):
        super(dsac, self).__init__()

        self.convolution1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=(1, 1)),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=128)

        )

        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))

        self.attention_1 = Att.SelfAttention(101, 4, 0.2)
        self.attention_2 = Att.SelfAttention(42, 128, 0.7)

        self.convolution2 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), stride=(1, 1)),
        )


        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )





    def _forward_impl(self, seq):
        seq = seq.float()

        seq = seq.squeeze(1)
        att_seq_1 = self.attention_1(seq)
        att_seq_1 = att_seq_1.unsqueeze(1)

        conv_seq_1 = self.convolution1(att_seq_1)#seq
        pool_seq_1 = self.max_pooling_1(conv_seq_1)

        att_seq_2 = pool_seq_1.squeeze(2)
        att_seq_2 = self.attention_2(att_seq_2)
        att_seq_2 = att_seq_2.unsqueeze(2)

        conv_seq_2 = self.convolution2(pool_seq_1)
        att_seq_2 = self.convolution2(att_seq_2)
        #print(att_seq_2.shape)

        return self.output(att_seq_2),self.output(conv_seq_2)
    def forward(self, seq):
        return self._forward_impl(seq)