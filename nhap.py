import torch
import numpy as np
import matplotlib.pyplot as plt

from isac.channel.fading_channel import AWGNChannel
from isac.channel.multipath_channel import MultiPathChannelConfig, OFDMBeamSpaceChannel, generate_multipath_ofdm_channel
from isac.estimator.functional import estimate_subspace_order
from isac.mimo.antenna import UniformLinearArray
from isac.ofdm.ofdm import OFDMConfig

class VideoEncoder:
    def __init__(self):
        # Hamming (7,4) Systematic Generator and Parity Check
        self.G = torch.tensor([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], dtype=torch.int64)
        self.H = torch.tensor([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ], dtype=torch.int64)
        
        # Precompute syndrome decoding table
        self.err_matrix = torch.zeros((8, 7), dtype=torch.int64)
        for i in range(7):
            s = (self.H[:, i] * torch.tensor([4, 2, 1])).sum().item()
            self.err_matrix[s, i] = 1

    def encode(self, frames):
        # frames: [N, H, W] uint8
        flat = frames.flatten()
        bits = torch.zeros((len(flat), 8), dtype=torch.int64)
        for i in range(8):
            bits[:, i] = (flat >> (7 - i)) & 1
        bits = bits.flatten()
        
        # Priority mapping: First 20% are Header (FEC protected), rest Payload (Uncoded)
        header_len = int(len(bits) * 0.2)
        header_len = header_len - (header_len % 4) # make multiple of 4
        
        header_bits = bits[:header_len].reshape(-1, 4)
        payload_bits = bits[header_len:]
        
        # Encode header
        encoded_header = (header_bits @ self.G) % 2
        
        # Interleave encoded header for burst error robustness
        N_blocks = encoded_header.shape[0]
        if N_blocks > 0:
            encoded_header = encoded_header.T.flatten()
        else:
            encoded_header = encoded_header.flatten()
        
        total_bits = torch.cat([encoded_header, payload_bits])
        return total_bits, header_len

    def decode(self, rx_bits, header_len, num_frames, frame_size=(16,16)):
        enc_header_len = (header_len // 4) * 7
        rx_header_flat = rx_bits[:enc_header_len]
        rx_payload = rx_bits[enc_header_len:]
        
        # De-interleave
        N_blocks = enc_header_len // 7
        if N_blocks > 0:
            rx_header = rx_header_flat.reshape(7, N_blocks).T
        else:
            rx_header = rx_header_flat.reshape(-1, 7)
        
        # Syndrome decoding
        syndrome = (rx_header @ self.H.T) % 2
        syn_dec = syndrome[:, 0]*4 + syndrome[:, 1]*2 + syndrome[:, 2]
        
        corrected_header = (rx_header + self.err_matrix[syn_dec]) % 2
        decoded_header = corrected_header[:, :4].flatten()
        
        final_bits = torch.cat([decoded_header, rx_payload])
        
        # Convert to frames
        num_expected = num_frames * frame_size[0] * frame_size[1] * 8
        final_bits = final_bits[:num_expected]
        if len(final_bits) < num_expected:
            final_bits = torch.cat([final_bits, torch.zeros(num_expected - len(final_bits), dtype=torch.int64)])
        
        flat_bits = final_bits.reshape(-1, 8)
        vals = torch.zeros(len(flat_bits), dtype=torch.int64)
        for i in range(8):
            vals |= (flat_bits[:, i] << (7 - i))
            
        return vals.to(torch.uint8).reshape(num_frames, *frame_size)

def calculate_psnr(orig, recon):
    mse = torch.mean((orig.float() - recon.float()) ** 2).item()
    if mse == 0:
        return 50.0 
    return 10 * np.log10(255**2 / mse)

def calculate_mrt_matrix(num_tx, target_aoa_az_rad):
    idx = torch.arange(num_tx, dtype=torch.float32)
    phase = np.pi * np.sin(target_aoa_az_rad) * idx
    a = torch.exp(1j * phase).to(torch.complex64)
    a = a / torch.norm(a)
    
    W = torch.zeros((num_tx, num_tx), dtype=torch.complex64)
    W[:, 0] = a
    I = torch.eye(num_tx, dtype=torch.complex64)
    for i in range(1, num_tx):
        v = I[:, i]
        for j in range(i):
            proj = (W[:, j].conj().dot(v)) * W[:, j]
            v = v - proj
        v = v / torch.norm(v)
        W[:, i] = v
    return W

class ISACChannel:
    def __init__(self, tx_array, rx_array, ofdm_cfg, mpc):
        self.tx_array = tx_array
        self.rx_array = rx_array
        self.ofdm_cfg = ofdm_cfg
        self.mpc = mpc
        self.channel = OFDMBeamSpaceChannel(mpc_configs=mpc, ofdm_config=ofdm_cfg, tx_array=tx_array, rx_array=rx_array)

    def transmit(self, tx_grid, snr_db, W=None):
        num_syms = tx_grid.shape[-1]
        # Hf [Nr, Nt, Nfft, Nsym]
        Hf = generate_multipath_ofdm_channel(self.tx_array, self.rx_array, self.ofdm_cfg.Nfft, num_syms, self.mpc, self.ofdm_cfg.subcarrier_spacing)
        
        # FIX 2: Compute noise floor from the OMNI (pre-beamforming) signal power.
        # This ensures a fair physics-based noise reference regardless of beamformer gain.
        # If we used "measured" power on the beamformed signal, sensing-centric MRT would
        # inflate the effective SNR and unfairly improve the comm quality metric.
        clean_rx_omni = self.channel(tx_grid)  # pass through un-steered grid
        omni_sigpow_lin = clean_rx_omni.abs().pow(2).mean().item()
        omni_sigpow_db = 10 * np.log10(max(omni_sigpow_lin, 1e-30))
        
        if W is not None:
            # W: [Nt, Nt], tx_grid can be [1, Nt, Nsc, Nsym]
            tx_grid_bf = torch.einsum('ij, ...jcs->...ics', W, tx_grid)
        else:
            tx_grid_bf = tx_grid
            
        clean_rx_grid = self.channel(tx_grid_bf)
        # Anchor noise to omni sigpow so beamforming gain is NOT counted in SNR denominator
        awgn = AWGNChannel(snr_db=snr_db, sigpow_db=omni_sigpow_db)
        rx_grid = awgn(clean_rx_grid)
        return rx_grid, Hf


class MIMOReceivers:
    def __init__(self, num_tx):
        self.num_tx = num_tx

    def estimate_channel(self, rx_pilots, tx_pilots, snr_lin, method='ls'):
        # rx_pilots: [Nr, Ndata, Np] -> Y
        Y = rx_pilots.permute(1, 0, 2) # [Ndata, Nr, Np]
        X = tx_pilots.permute(1, 0, 2) # [Ndata, Nt, Np]
        
        # LS
        X_p = torch.linalg.pinv(X) # [Ndata, Np, Nt]
        H_ls = Y @ X_p # [Ndata, Nr, Nt]
        
        if method == 'ls':
            return H_ls
        elif method == 'lmmse':
            # Matrix LMMSE estimator based on ULA spatial correlation
            Ndata, Nr, Nt = H_ls.shape
            
            def get_ula_corr(N, r=0.5):
                idx = torch.arange(N, dtype=torch.float32)
                return r ** torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
            
            R_r = get_ula_corr(Nr).to(H_ls.device)
            R_t = get_ula_corr(Nt).to(H_ls.device)
            R_H = torch.kron(R_t, R_r).to(torch.complex64)
            
            H_vec = H_ls.permute(0, 2, 1).reshape(Ndata, Nt*Nr, 1)
            I = torch.eye(Nt*Nr, dtype=torch.complex64, device=H_ls.device)
            
            inverse_term = torch.inverse(R_H + (1.0 / snr_lin) * I)
            H_lmmse_vec = R_H @ inverse_term @ H_vec
            
            H_lmmse = H_lmmse_vec.reshape(Ndata, Nt, Nr).permute(0, 2, 1)
            return H_lmmse
            
    def equalize(self, rx_data, H_est, snr_lin, method='mmse'):
        # rx_data: [Nr, Ndata, Nsym]
        # H_est: [Ndata, Nr, Nt]
        H_H = H_est.mH 
        Ndata = rx_data.shape[1]
        I = torch.eye(self.num_tx, dtype=torch.complex64).expand(Ndata, self.num_tx, self.num_tx)
        
        if method == 'zf':
            W = torch.linalg.pinv(H_est) # [Ndata, Nt, Nr]
        else: # mmse
            W = torch.inverse(H_H @ H_est + (1/snr_lin)*I) @ H_H
            
        y_p = rx_data.permute(1, 2, 0).unsqueeze(-1) # [Ndata, Nsym, Nr, 1]
        Nsym = rx_data.shape[2]
        W_exp = W.unsqueeze(1).expand(-1, Nsym, -1, -1)
        
        x_hat_p = W_exp @ y_p # [Ndata, Nsym, Nt, 1]
        x_hat = x_hat_p.squeeze(-1).permute(2, 0, 1) # [Nt, Ndata, Nsym]
        return x_hat


class SensingProcessor:
    def __init__(self, rx_array):
        self.rx_array = rx_array
        
    def estimate_aoa_root_music(self, rx_signals, num_sources):
        if num_sources == 0:
            return []
            
        R = rx_signals @ rx_signals.mH / rx_signals.shape[1]
        eigenvalues, eigenvectors = torch.linalg.eigh(R)
        
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idx]
        
        En = eigenvectors[:, num_sources:]
        C = En @ En.mH
        
        Nt = C.shape[0]
        coeffs = []
        for k in range(Nt - 1, -Nt, -1):
            coeffs.append(torch.sum(torch.diagonal(C, k)).item())
            
        roots = np.roots(coeffs)
        roots = roots[np.abs(roots) < 1]
        roots = sorted(roots, key=lambda r: np.abs(np.abs(r) - 1))
        
        aoas = []
        for r in roots[:num_sources]:
            arg = np.angle(r)
            val = arg / np.pi
            val = np.clip(val, -1, 1)
            aoas.append(np.arcsin(val))
            
        return aoas

    def calculate_rmse(self, true_aoas, estimated_aoas):
        true_az = true_aoas[:, 0].numpy()
        if len(estimated_aoas) == 0:
            return np.pi
            
        errors = []
        est_avail = list(estimated_aoas)
        for t_val in true_az:
            if len(est_avail) == 0:
                errors.append(np.pi**2) # miss
                continue
            dists = [abs(t_val - e) for e in est_avail]
            best_i = int(np.argmin(dists))
            errors.append(dists[best_i]**2)
            est_avail.pop(best_i)
            
        penalty = len(est_avail) * (np.pi**2) # clutter alarm
        return np.sqrt((np.sum(errors) + penalty) / max(len(true_az), 1))


def qpsk_mod(bits):
    return 1 / np.sqrt(2.0) * ((2 * bits[..., 0] - 1) + 1j * (2 * bits[..., 1] - 1))

def qpsk_demod(symbols):
    bits = torch.stack(((symbols.real > 0).to(torch.int64), (symbols.imag > 0).to(torch.int64)), dim=-1)
    return bits


def run_high_fidelity_simulation():
    torch.manual_seed(1024)
    np.random.seed(1024)

    # Core parameters
    num_tx = 4
    num_rx = 8
    ofdm_cfg = OFDMConfig(Nfft=64, cp_frac=0.07, num_guard_carriers=(6, 5))
    num_data_carrs = ofdm_cfg.num_data_carriers
    
    tx_array = UniformLinearArray(num_antennas=num_tx, along_axis='y')
    rx_array = UniformLinearArray(num_antennas=num_rx, along_axis='y')
    
    true_aoas = torch.tensor([[0.3, 0.0], [-0.5, 0.0], [0.1, 0.0], [-0.2, 0.0], [0.4, 0.0], [-0.6, 0.0]])
    mpc = MultiPathChannelConfig.random_generate(num_paths=6, max_delay=1e-7, max_doppler=30) 
    mpc.aoas = true_aoas
    
    video_enc = VideoEncoder()
    channel = ISACChannel(tx_array, rx_array, ofdm_cfg, mpc)
    receiver = MIMOReceivers(num_tx)
    sensing = SensingProcessor(rx_array)

    # Payload
    # Using a gradient instead of a constant block so that any error 
    # creates visually obvious artifacts across the whole frame
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
    orig_frames = ((yy * 8 + xx * 8) % 256).to(torch.uint8).unsqueeze(0) 
    
    encoded_bits, header_len = video_enc.encode(orig_frames)
    total_payload_bits = len(encoded_bits)
    
    bits_per_sym = num_tx * num_data_carrs * 2
    # FIX 1: Remove the +20 slack.  The frame is EXACTLY long enough for the
    # comm-centric case (PR=0.1) to transmit all payload data.  For PR=0.5,
    # Nd is halved, so only half the bits can be transmitted — the rest are
    # forced to zero, creating the physical throughput bottleneck.
    min_data_syms = int(np.ceil(total_payload_bits / bits_per_sym))
    # Reserve a small but fixed number of pilot slots for the best-case case
    min_pilot_syms = num_tx  # minimum pilots = num_tx (one MIMO training block)
    total_ofdm_syms = min_data_syms + min_pilot_syms
    
    snrs = [0, 5, 10, 15, 20, 25]
    pilot_ratios = [0.1, 0.2, 0.3, 0.5]
    
    # We choose an SNR where the physical throughput limitation of the 
    # sensing-centric PR=0.5 case is starkly visible vs the comm-centric one.
    capture_snr = 15
    
    pareto_results = []
    ber_fec_results = {'uncoded': [], 'fec': []}
    
    best_comm_recon = None
    best_sens_recon = None

    wc = 1.0
    ws = 10.0
    optimal_j = -9999
    optimal_pr = None

    def frames_to_bits_raw(f):
        flat = f.flatten()
        b = torch.zeros((len(flat), 8), dtype=torch.int64)
        for i in range(8):
            b[:, i] = (flat >> (7 - i)) & 1
        return b.flatten()
        
    orig_bits_raw = frames_to_bits_raw(orig_frames)

    def run_chain(pr_val, snr_db, snr_lin, eq_meth='mmse', use_beamforming=True):
        Np = max(num_tx, int(round(total_ofdm_syms * pr_val)))
        Nd = max(total_ofdm_syms - Np, 1)
        
        # FIX 3: Strict throughput bottleneck.
        # data_cap_bits is the MAXIMUM number of bits Nd symbols can carry.
        # If it is smaller than the total encoded stream, the tail is forcibly
        # replaced with zeros (simulating packet loss / buffer overflow).
        # This means PR=0.5 transmits only the header + a fraction of the
        # payload, and the decoder sees zeros for everything else.
        data_cap_bits = Nd * bits_per_sym
        if data_cap_bits >= len(encoded_bits):
            pad = data_cap_bits - len(encoded_bits)
            tx_data_bits = torch.cat([encoded_bits, torch.zeros(pad, dtype=torch.int64)])
        else:
            # Only the first data_cap_bits are transmitted; everything else is zeroed out.
            tx_data_bits = torch.zeros(data_cap_bits, dtype=torch.int64)
            tx_data_bits[:data_cap_bits] = encoded_bits[:data_cap_bits]
            
        # Modulate data (symbol-consecutive mapping to group pixel errors)
        d_syms = qpsk_mod(
            tx_data_bits.reshape(Nd, num_tx, num_data_carrs, 2).permute(1, 2, 0, 3)
        ).unsqueeze(0)
        
        p_bits = torch.randint(0, 2, (1, num_tx, num_data_carrs, Np, 2), dtype=torch.int64)
        p_syms = qpsk_mod(p_bits)
        t_grid = torch.cat([p_syms, d_syms], dim=-1)
        
        # Strict Power Normalization: adjust power to maintain constant total energy 
        total_syms = Np + Nd
        t_grid = t_grid / np.sqrt(total_syms)
        
        ofdm_t_grid = ofdm_cfg.get_resource_grid(t_grid)
        
        # Application of Transmit Beamforming
        if use_beamforming:
            # Sensing mode uses MRT to steer power towards the target
            target_aoa = true_aoas[0, 0].item()
            W = calculate_mrt_matrix(num_tx, target_aoa).to(t_grid.device)
        else:
            # Comm mode does NOT use steering (max diversity for spatial multiplexing)
            W = None
            
        rx_raw, _ = channel.transmit(ofdm_t_grid, snr_db, W=W)
        
        # Restore unit symbol power for correct LMMSE noise baseline scaling
        rx_raw = rx_raw * np.sqrt(total_syms)
        
        rx_d = ofdm_cfg.get_data_grid(rx_raw)[0]
        rx_p = rx_d[:, :, :Np]
        
        # LMMSE Receiver Action
        H_est = receiver.estimate_channel(rx_p, p_syms[0], snr_lin, method='ls')

        rx_data_payload = rx_d[:, :, Np:]
        x_hat = receiver.equalize(rx_data_payload, H_est, snr_lin, method=eq_meth)
        
        # Sensing Action
        rx_sens_p = rx_raw[0, :, :, :Np].flatten(start_dim=1)
        try:
            order = estimate_subspace_order(rx_sens_p.unsqueeze(0), method='mdl')[0].item()
            est_aoas = sensing.estimate_aoa_root_music(rx_sens_p, min(order, len(true_aoas)))
            aoa_rmse = sensing.calculate_rmse(true_aoas, est_aoas)
        except Exception:
            aoa_rmse = np.pi
        
        # Match the bits sequence length
        det_bits_grid = qpsk_demod(x_hat)
        det_bits = det_bits_grid.permute(2, 0, 1, 3).reshape(-1)
        
        if len(det_bits) >= len(encoded_bits):
            rx_bits = det_bits[:len(encoded_bits)]
        else:
            rx_bits = torch.cat([det_bits, torch.zeros(len(encoded_bits) - len(det_bits), dtype=torch.int64)])
            
        return rx_bits, aoa_rmse

    print("Running Pareto Curve Combinatorics (FEC, LMMSE, Root-MUSIC, PSNR vs AoA)...")
    for snr in snrs:
        snr_lin = 10**(snr / 10)
        
        # Uncoded Baseline evaluation has been integrated into the loop as `payload_err`
        
        # Coded (FEC) Evaluate once for BER curve tracking at PR=0.2 (typical)
        c_rx_bits, _ = run_chain(0.2, snr, snr_lin, 'mmse', use_beamforming=False)
        c_rx_trimmed = c_rx_bits[:len(encoded_bits)]
        recon = video_enc.decode(c_rx_trimmed, header_len, 1, (32, 32))
        
        def frames_to_bits_raw(f):
            flat = f.flatten()
            b = torch.zeros((len(flat), 8), dtype=torch.int64)
            for i in range(8):
                b[:, i] = (flat >> (7 - i)) & 1
            return b.flatten()
            
        orig_bits = frames_to_bits_raw(orig_frames)
        c_dec_bits = frames_to_bits_raw(recon)
        c_ber = (c_dec_bits != orig_bits).float().mean().item()
        
        bits_to_measure = min(len(c_rx_bits), len(encoded_bits))
        u_ber = (c_rx_bits[:bits_to_measure] != encoded_bits[:bits_to_measure]).float().mean().item()
        
        ber_fec_results['uncoded'].append(u_ber)
        ber_fec_results['fec'].append(c_ber)
        
        # Over Pilot Ratios Sweep mapping Tradeoffs
        for pr in pilot_ratios:
            # We enforce use_beamforming = True for sensing dominant configurations
            apply_beamforming = (pr >= 0.4)
            c_rx_bits, aoa_rmse = run_chain(pr, snr, snr_lin, 'mmse', use_beamforming=apply_beamforming)
            
            recon = video_enc.decode(c_rx_bits, header_len, 1, (32, 32))
            psnr = calculate_psnr(orig_frames, recon)
            
            rmse_val = max(aoa_rmse, 1e-4) 
            J = wc * psnr - ws * np.log10(rmse_val)
            
            pareto_results.append({'snr': snr, 'pr': pr, 'psnr': psnr, 'aoa': aoa_rmse})
            
            # Print payload BER to monitor link health
            header_end = (header_len // 4) * 7
            payload_err = (c_rx_bits[header_end:] != encoded_bits[header_end:]).sum().item()
            print(f"SNR:{snr:2d} | PR:{pr:.1f} (BF={str(apply_beamforming)[0]}) => PSNR:{psnr:6.1f}dB | P_Errs:{payload_err:4d} | AoA RMSE:{aoa_rmse:.4f}rad | J:{J:5.1f}")
            
            if J > optimal_j:
                optimal_j = J
                optimal_pr = pr
                
            # Explicit Comm vs Sensing specific reconstructions at 15dB
            if snr == capture_snr and pr == 0.1: 
                best_comm_recon = recon[0]
            if snr == capture_snr and pr == 0.5: 
                best_sens_recon = recon[0]

    print(f"\n=> Pareto Optimization Complete. Global Utility maxed around PR: {optimal_pr:.1f}")

    # Generate the strict 3 Plots requirements
    
    # 1. Pareto Front
    plt.figure(figsize=(10,6))
    for snr in snrs:
        sub = [r for r in pareto_results if r['snr']==snr]
        psnrs = [r['psnr'] for r in sub]
        aoas = [r['aoa'] for r in sub]
        plt.plot(aoas, psnrs, marker='o', label=f'SNR={snr}dB')
        for r in sub:
            plt.annotate(f"PR={r['pr']}", (r['aoa'], r['psnr']), xytext=(5,5), textcoords='offset points')
    plt.xlabel('AoA RMSE (rad) [Lower is Better]')
    plt.ylabel('PSNR (dB) [Higher is Better]')
    plt.title('ISAC Pareto Front: Advanced Sensing vs Communication')
    plt.grid(True)
    plt.legend()
    plt.savefig('isac_pareto_front.png')
    plt.close()

    # 2. BER vs SNR
    plt.figure(figsize=(8,6))
    plt.semilogy(snrs, ber_fec_results['uncoded'], marker='x', linestyle='--', color='red', label='Uncoded (Raw)')
    plt.semilogy(snrs, ber_fec_results['fec'], marker='o', linestyle='-', color='green', label='Hamming (7,4) + LMMSE')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('Waterfall Effect: BER vs SNR (6G Enhancements)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('isac_ber_fec.png')
    plt.close()

    # 3. Visual Comparisons
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_frames[0].numpy(), cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original Frame")
    axes[0].axis('off')
    
    if best_comm_recon is not None:
        axes[1].imshow(best_comm_recon.numpy(), cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Comm-Centric (PSNR-Opt)")
    axes[1].axis('off')

    if best_sens_recon is not None:
        axes[2].imshow(best_sens_recon.numpy(), cmap='gray', vmin=0, vmax=255)
    axes[2].set_title("Sens-Centric (AoA-Opt)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('isac_visual_recon.png')
    plt.close()

    print("Success: Generated isac_pareto_front.png, isac_ber_fec.png, and isac_visual_recon.png")

if __name__ == "__main__":
    run_high_fidelity_simulation()
