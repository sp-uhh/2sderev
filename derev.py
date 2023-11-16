import torch
import argparse
import os
from os.path import join
import soundfile as sf
import json
import glob

from dnn import backbones
from dsp import wpe, pf

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

class Dereverberator():

    def __init__(self, params):
        self.F = params["n_fft"] // 2 + 1
        self.channels = params["channels"]

        self.wpe = getattr(wpe, params["wpe"]["class"])(params, params["wpe"])
        self.dnn_wpe = getattr(backbones, params["dnn_wpe"]["class"])(**params["dnn_wpe"])
        self.dnn_wpe.load_state_dict(torch.load(params["dnn_wpe"]["ckpt"]))
        self.pf = getattr(pf, params["pf"]["class"])(params, params["pf"])
        self.dnn_pf = getattr(backbones, params["dnn_pf"]["class"])(**params["dnn_pf"])
        self.dnn_pf.load_state_dict(torch.load(params["dnn_pf"]["ckpt"]))

        self.dnn_wpe.load_stats(params["dnn_wpe"]["stats"])
        self.dnn_pf.load_stats(params["dnn_pf"]["stats"])

    def to(self, device):
        self.device = device
        self.wpe = self.wpe.to(device)
        self.dnn_wpe = self.dnn_wpe.to(device)
        self.pf = self.pf.to(device)
        self.dnn_pf.to(device)
        return self

    def reset_state(self):
        self.dnn_wpe.reset_state()
        self.wpe.reset_state()
        self.dnn_pf.reset_state()
        self.pf.reset_state()

    def process(self, Y, **kwargs):
        F, D, T = Y.size()

        assert self.F == F, f"Mismatch in frequency bins : {self.F} model vs {F} input"
        assert self.channels == D, f"Mismatch in channels : {self.channels} model vs {D} input"

        X = torch.zeros_like(Y) #[F, D, T]
        with torch.no_grad():
            for _ in range(2): #initialize WPE statistics to get the best performance (optional)
                for t in range(T):
                    Y_update = Y[..., t] #F,D

                    # DNN-WPE
                    clean_mag, _ = self.dnn_wpe(Y_update.unsqueeze(-1), return_interference=False) #F,1,1
                    clean_periodogram = torch.square(clean_mag.squeeze()) #F
                    # WPE
                    X[..., t] = self.wpe.step_online(Y_update, clean_periodogram.squeeze()) #[F, D]
                    # DNN-PF
                    speech_mag, interference_mag = self.dnn_pf(X[..., t].unsqueeze(-1), singlechannel=False, return_interference=True) #F,D,1
                    speech_periodogram, interference_periodogram = torch.square(speech_mag.squeeze()), torch.square(interference_mag.squeeze()) #F,D
                    # PF
                    X[..., t] = self.pf.step_online(X[..., t], speech_periodogram, interference_periodogram) #[F, D]

        X[: 2] = .0 + 0j
        return X.cpu()



if __name__ == "__main__":

    with open("derev_params.json", "r") as j:
        params_dict = json.load(j)

    parser = argparse.ArgumentParser()
    parser.add_argument('--speech', type=str, help="Input multi-channel file")
    parser.add_argument('--config', type=str, choices=list(params_dict.keys()), 
        default='wpe+pf_ha', help="Choice of parameterization from derev_params.json")
    args = parser.parse_args()

    params = params_dict[args.config]

    print("Dereverberating file/dir...")

    if params["window_type"] == "hann":
        window = torch.hann_window(params["n_fft"])
    elif params["window_type"] == "sqrt_hann":
        window = torch.sqrt(torch.hann_window(params["n_fft"]))
    else:
        raise NotImplementedError
    
    istft_kwargs = {
        "n_fft": params["n_fft"],
        "hop_length": params["hop_length"],
        "window": window,
        "center": True
    }
    stft_kwargs = {
        **istft_kwargs,
        "return_complex": True
    }

    if os.path.isdir(args.speech):
        speech = sorted(glob.glob(join(args.speech, "*.wav")))
    else:
        speech = [args.speech]

    dereverberator = Dereverberator(params)

    for speech_path in speech:
        
        dereverberator.reset_state()
        dereverberator = dereverberator.to(torch.device("cuda:0"))

        y, sr = sf.read(speech_path) #d,t
        Y = torch.stft(torch.from_numpy(y).transpose(0, 1), **stft_kwargs) #d,f,t
        Y = Y.permute(1, 0, 2)
        X = dereverberator.process(Y.to(dereverberator.device))
        X = X.permute(1, 0, 2)
        x = torch.istft(X, **istft_kwargs).transpose(0, 1).numpy()

        ensure_dir(join("results", params["name"]))
        sf.write(join("results", params["name"], os.path.basename(speech_path)), x, sr)