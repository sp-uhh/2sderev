import torch
# from scipy.ss import gamma, hyp1f1

class PostFilterAbstract():

    def __init__(self, params, pf_params):
        self.freq_bins = params["n_fft"] // 2 + 1
        self.channels = params["channels"]
        for key, value in pf_params.items():
            setattr(self, key, value)

    def to(self, device):
        pass

    def reset_state(self, *args, **kwargs):
        pass

    def step_online(self, Y_update, *args, **kwargs):
        pass



class Bypass(PostFilterAbstract):

    def __init__(self, params, pf_params):
        super().__init__(params, pf_params)

    def to(self, device):
        return self

    def reset_state(self):
        pass

    def step_online(self, Y_update, *args, **kwargs):
        return Y_update




class DirectMasker(PostFilterAbstract):

    def __init__(self, params, pf_params):
        super().__init__(params, pf_params)

    def to(self, device):
        return self

    def reset_state(self):		
        pass

    def step_online(self, Y_update, clean_periodogram, *args, **kwargs):
        processed = torch.sqrt(clean_periodogram).type(Y_update.dtype) * torch.exp(1j*Y_update.angle())
        return self.bleeding * Y_update + (1-self.bleeding) * processed



class WienerFilter(PostFilterAbstract):

    def __init__(self, params, pf_params):
        super().__init__(params, pf_params)
        self.eps_pf = 1e-8

    def to(self, device):
        self.clean_psd = self.clean_psd.to(device)
        self.interference_psd = self.interference_psd.to(device)
        return self

    def reset_state(self):
        self.clean_psd = torch.zeros(self.freq_bins, self.channels)
        self.interference_psd = torch.zeros(self.freq_bins, self.channels)

    def step_online(self, Y_update, clean_periodogram, interference_periodogram, *args, **kwargs):
        """
        Args:
            - Y: [F, D]
            - clean/interference_periodogram: [F]
        Returns
            - speech [F, D] complex
        """
        self.clean_psd = self.alpha_s * self.clean_psd + (1-self.alpha_s) * clean_periodogram
        self.interference_psd = self.alpha_s * self.interference_psd + (1-self.alpha_s) * interference_periodogram

        WFGain = self.clean_psd / (self.clean_psd + self.interference_psd + self.eps_pf)
        WFGain = torch.max( WFGain, 10**(self.gmin/10) * torch.ones_like(self.clean_psd) )
        
        processed = WFGain * Y_update
        return self.bleeding * Y_update + (1-self.bleeding) * processed


class SuperGaussianEstimator(PostFilterAbstract):

    def __init__(self, params, pf_params):
        super().__init__(params, pf_params)
        self.eps_pf = 1e-8
        raise NotImplementedError

    def reset_state(self):
        self.clean_psd = torch.zeros(self.freq_bins, self.channels)
        self.interference_psd = torch.zeros(self.freq_bins, self.channels)

    def to(self, device):
        self.clean_psd = self.clean_psd.to(device)
        self.interference_psd = self.interference_psd.to(device)
        return self

    def step_online(self, Y_update, clean_periodogram, interference_periodogram, *args, **kwargs):
        Y_phase = torch.angle(Y_update)

        self.clean_psd = self.alpha_s * self.clean_psd + (1-self.alpha_s) * clean_periodogram
        self.interference_PSD = self.alpha_n * self.interference_psd + (1-self.alpha_n) * interference_periodogram
        aprioriSNR = self.clean_PSD / (self.interference_PSD + self.eps_pf)
        aposterioriSNR = torch.square(torch.abs(Y_update)) / (self.interference_psd + self.eps_pf)

        pre = aprioriSNR / (self.mu + aprioriSNR)
        nu = pre * aposterioriSNR

        gamma_factor = gamma(self.mu + self.beta/2)/gamma(self.mu)
        phi_factor = torch.from_numpy( hyp1f1(1 - self.mu - self.beta/2, 1, -nu.cpu().numpy()) ).to(self.device) / \
            ( self.eps_pf + torch.from_numpy( hyp1f1(1 - self.mu, 1, -nu.cpu().numpy()) ) ).to(self.device)
        
        X_new = ( torch.pow(gamma_factor * phi_factor, 1/self.beta) * torch.sqrt( pre ) * torch.sqrt(self.interference_psd) ) * torch.exp(1j * Y_phase)
        
        return X_new