import torch

class WPEAbstract():

    def __init__(self, params, wpe_params):
        self.freq_bins = params["n_fft"] // 2 + 1
        self.channels = params["channels"]
        for key, value in wpe_params.items():
            setattr(self, key, value)

    def to(self, device):
        pass

    def reset_state(self, *args, **kwargs):
        pass

    def step_online(self, Y_update, *args, **kwargs):
        pass


class Bypass(WPEAbstract):

    def to(self, device):
        return self

    def reset_state(self, *args, **kwargs):
        pass

    def step_online(self, Y_update, *args, **kwargs):
        return Y_update


class RLSWPE(WPEAbstract):
        
    def __init__(self, params, wpe_params):
        super().__init__(params, wpe_params)
        self.dtype = torch.complex128

    def reset_state(self):
        self.Y_buffer = torch.zeros(self.freq_bins, self.channels, self.delay+1).type(self.dtype)
        self.Y_tilde = torch.zeros(self.freq_bins, self.channels*self.taps).type(self.dtype) #[freq_bins, channels*taps]
        self.inv_R_WPE = torch.eye(self.channels*self.taps).unsqueeze(0).type(self.dtype).expand(self.freq_bins, self.channels*self.taps, self.channels*self.taps) #[freq_bins, channels*taps, channels*taps]
        self.G_WPE = torch.zeros(self.freq_bins, self.channels*self.taps, self.channels).type(self.dtype)

    def to(self, device):
        self.Y_buffer = self.Y_buffer.to(device)
        self.Y_tilde = self.Y_tilde.to(device)
        self.inv_R_WPE = self.inv_R_WPE.to(device)
        self.G_WPE = self.G_WPE.to(device)
        return self

    def _update_buffer(self, Y_buffer, Y_update):
        """Returns the updated buffer to track the corresponding t-Delta frame
            Y_buffer [(b), F, D, delay+1]
            Y_update: [(b), F, D]
            Y_buffer_k [(b), F, D, delay+1]
        """
        return torch.cat([Y_buffer[..., 1: ], Y_update.unsqueeze(-1)], dim=-1)

    def _update_Y_tilde_WPE(self, Y_tilde, Y_update_delayed):
        """
            Y_tilde [(b), freq_bins, channels*taps] or [(b), freq_bins, channels**2*taps, channels]
            Y_update_delayed: [(b), freq_bins, channels] containing y_{t-delay}
            Y_tilde_k [(b), freq_bins, channels*taps]
        """
        return torch.cat([Y_tilde[..., self.channels:], Y_update_delayed], dim=-1)

    def _update_kalman_gain_WPE(self, inv_R, Y_tilde, psd):
        """
            inv_R [(b), freq_bins, channels*taps, channels*taps]
            Y_tilde [(b), freq_bins, channels*taps]
            psd [(b), freq_bins]
            K [(b), freq_bins, channels*taps]
        """
        numerator = torch.mul(1 - self.alpha, torch.matmul(inv_R, Y_tilde.unsqueeze(-1).conj())) #[(b), freq_bins, channels*taps, 1]
        denominator = torch.add( self.alpha * psd.unsqueeze(-1), torch.matmul(Y_tilde.unsqueeze(-2), numerator)[:, 0]) #[(b), freq_bins, 1]
        
        return torch.mul( 1 / torch.add(denominator, self.eps_wpe), numerator[..., 0]) #[(b), freq_bins, channels*taps]

    def _update_inv_cov_WPE(self, inv_R, K, Y_tilde):
        """
            inv_R [(b), freq_bins, channels*tap, channels*taps] if method_name="online_wpe" [freq_bins, channels**2*taps, channels**2*taps] if method_name="online_gwpe"
            K [(b), freq_bins, channels*taps]  if method_name="online_wpe" [freq_bins, channels**2*taps, channels] if method_name="online_gwpe"
            Y_tilde [(b), freq_bins, channels*taps]
            inv_R_k [(b), freq_bins, channels*tap, channels*taps]  if method_name="online_wpe" [freq_bins, channels**2*taps, channels**2*taps]if method_name="online_gwpe"
        """
        return 1/self.alpha * torch.add(inv_R, - torch.matmul(K.unsqueeze(-1), torch.matmul(Y_tilde.unsqueeze(-2), inv_R)))

    def _update_taps_WPE(self, X, G, K):
        """
            X [(b), freq_bins, channels, 1] or X_d [freq_bins, 1] if channelwise
            G [(b), freq_bins, channels*taps, channels] or G_d [freq_bins, channels*taps] if channelwise 
            K [(b), freq_bins, channels*taps]
            G_k [freq_bins, channels*taps, channels] or G_k_d [freq_bins, channels*taps] if channelwise
        """
        return torch.add(G, torch.matmul(K.unsqueeze(-1), X.transpose(-2, -1))) #[(b), freq_bins, channels*taps, channels] 

    def _update_estimate_WPE(self, Y_update, G_WPE, Y_tilde):
        return torch.add(Y_update.unsqueeze(-1), - torch.matmul(G_WPE.transpose(-1, -2), Y_tilde.unsqueeze(-1)))

    

    def step_online(self, Y_update, clean_periodogram, **kwargs):
        """
            @args:
                - Y_update [F, D]
                - clean_periodogram [F]
            @returns:
                - X_new [F, D]
        """
        
        self.Y_buffer = self._update_buffer(self.Y_buffer, Y_update)
        Y_update_delayed = self.Y_buffer[..., 0] #Frame y_{t-delay} #[F, D]
        self.Y_tilde = self._update_Y_tilde_WPE(self.Y_tilde, Y_update_delayed) #[F, D*taps]

        psd = clean_periodogram #No recursive averaging here
        K = self._update_kalman_gain_WPE(self.inv_R_WPE, self.Y_tilde, psd) #[f, d*k]
        self.inv_R_WPE = self._update_inv_cov_WPE(self.inv_R_WPE, K, self.Y_tilde) #[f, d*k, d*k]
        X_inter = self._update_estimate_WPE(Y_update, self.G_WPE, self.Y_tilde)  #[f, d, 1]
        self.G_WPE = self._update_taps_WPE(X_inter, self.G_WPE, K) #[f, d*k, d]
        X_new = self._update_estimate_WPE(Y_update, self.G_WPE, self.Y_tilde).squeeze()  #[f, d]

        return X_new
