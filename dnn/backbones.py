import pytorch_lightning as pl 
import torch
from argparse import ArgumentParser
import os 
from os.path import join

class Bypass(torch.nn.Module):

    def __init__(self, **ignored_kwargs):
        super().__init__()
    
    def reset_state(self):
        pass

    def load_state_dict(self, ckpt):
        pass

    def load_stats(self, *args, **kwargs):
        pass
        
    def forward(self, Y, *args, return_interference=False, **kwargs):
        if return_interference:
            return Y, Y
        else:
            return Y, None

    def to(self, device):
        return self



class LSTMNet(torch.nn.Module):

    def __init__(self,
                F: int = 257,
                n_lstm_hidden: int = 512,
                n_lstm_layers: int = 1,
                **ignored_kwargs):
        super().__init__()

        assert n_lstm_hidden % 2 == 0, f"Number of hidden units in LSTM has to be even but was {n_lstm_hidden}"
        self.F = F
        self.n_lstm_hidden = n_lstm_hidden
        self.n_lstm_layers = n_lstm_layers

        self.lstm_layers = torch.nn.LSTM(
            input_size=self.F,
            hidden_size=self.n_lstm_hidden,
            batch_first=True,
            num_layers=1
        )
        self.clean_map = torch.nn.Linear(self.n_lstm_hidden, self.F, bias=False)
        self.interference_map = torch.nn.Linear(self.n_lstm_hidden, self.F, bias=False)

        self.reset_state()

    def load_stats(self, stats_path):
        self.mean = torch.load(stats_path.format("mean")).unsqueeze(0).unsqueeze(-1) #1,F,1
        self.std = torch.load(stats_path.format("std")).unsqueeze(0).unsqueeze(-1) #1,F,1

    def get_z_score(self, x_mag):
        """
        x_mag is b,F,1,T or b,F,D,T
        z should be F,T,b
        """
        z = (x_mag[..., 0, :] - self.mean)/self.std #b,F,T
        return z.type(torch.float32) #b,F,T

    def reset_state(self):
        self.state = None

    def forward(self, x, singlechannel=True, return_interference=False):
        """
        x: #(b),F,D,T
        state: #( h_0=[n_layers, T, n_hidden], c_0=[n_layers, T, n_hidden])
        """
        x_mag = x[..., 0, :].clone().abs().unsqueeze(2) if singlechannel else x.clone().abs() #b,F,1,T or b,F,D,T
        if x_mag.ndimension() < 3: #lacks the batch size
            x_mag = x_mag.unsqueeze(0) #b,F,D,T
        z = self.get_z_score(x_mag).transpose(1, 2) #b,T,F

        output_lstm, self.state = self.lstm_layers(z, self.state) #input - output is [b,T,F]

        clean_map = self.clean_map(output_lstm) #b,T,F
        clean_mask = torch.sigmoid(clean_map) #b,T,F
        clean_mask = clean_mask.transpose(1, 2).unsqueeze(2) #b,F,1,T
        clean_mag = torch.mul(x_mag, clean_mask) #b,F,1,T or b,F,D,T

        if return_interference:
            interference_map = self.interference_map(output_lstm) #b,T,F
            interference_mask = torch.sigmoid(interference_map) #b,T,F
            interference_mask = interference_mask.transpose(1, 2).unsqueeze(2) #b,F,1,T
            interference_mag = torch.mul(x_mag, interference_mask) #b,F,1,T or b,F,D,T
        else:
            interference_mag = None

        return clean_mag, interference_mag

    def to(self, device):
        self = super().to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    # def load_from_checkpoint(self, checkpoint):
    #     # print(checkpoint["state_dict"].keys())
    #     new_state_dict = {}
    #     for key, value in checkpoint["state_dict"].items():
    #         shortened_key = key.split("subnet.")[-1]
    #         new_state_dict[shortened_key] = value
    #     checkpoint["state_dict"] = new_state_dict   
    #     # torch.save(checkpoint, "models/SimpleLSTM_KL_noLR_b128_CI_kemar_epoch=380.ckpt")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--ckpt")
    args = parser.parse_args()

    model = LSTMNet.load_from_checkpoint(args.ckpt)