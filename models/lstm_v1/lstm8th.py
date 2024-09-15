import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
import nvtx
from torch.nn.functional import silu
from typing import List

"""
Contains the code for the Unet and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class LSTM8thMetaData(modulus.ModelMetaData):
    name: str = "LSTM8th"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = False

class LSTM8th(modulus.Module):
    def __init__(self,
                 input_size,
                 seq_len,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=.1,
                 hidden_layers=[128, 256],
                 input_profile_vars=4):

        super().__init__(meta=LSTM8thMetaData())
        self.input_size = input_size
        self.input_profile_vars = input_profile_vars # t,q,u,v as input
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional=bidirectional
        self.output_size=output_size
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=0.1)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2 if bidirectional else hidden_size,
                                               num_heads=8,
                                               batch_first=True)

        if hidden_layers and len(hidden_layers):
            first_layer  = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_layers[0])
            self.hidden_layers = nn.ModuleList(
                [first_layer] + \
                [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers:
                nn.init.kaiming_normal_(layer.weight.data)
            # self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)
        else:
            self.hidden_layers = []
            # self.intermediate_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, self.input_size)
            self.output_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.activation_fn = torch.nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print('x:', x.shape)
        x_profile = x[:,:self.input_profile_vars*26]
        x_scalar = x[:,self.input_profile_vars*26:]
        print('x_profile:', x_profile.shape)
        print('x_scalar:', x_scalar.shape)
        x_profile = x_profile.reshape(-1, self.input_profile_vars, 26)
        x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, 26)
        x = torch.cat((x_profile, x_scalar), dim=1)
        print('x at 2:', x.shape)
        x = x.permute(0, 2, 1)
        
        # batch_size = x.size(0)
        outputs, hidden = self.rnn(x)
        print('outputs:', outputs.shape)
        outputs = outputs.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        attn_output, _ = self.attention(outputs, outputs, outputs)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)

        x = self.dropout(self.activation_fn(attn_output))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        print('x at 3:', x.shape)
        # (-1,26,4) -> (-1,102)
        x = x.permute(0,2,1).reshape(-1,102)
        return x