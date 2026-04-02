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
        
        if W is not None:
            # W: [Nt, Nt], tx_grid can be [1, Nt, Nsc, Nsym]
            tx_grid = torch.einsum('ij, ...jcs->...ics', W, tx_grid)
            
        clean_rx_grid = self.channel(tx_grid)
        awgn = AWGNChannel(snr_db=snr_db)
        rx_grid = awgn(clean_rx_grid)
        return rx_grid, Hf


class MIMOReceivers:
    def __init__(self, num_tx):
        self.num_tx = num_tx

    def estimate_channel(self, rx_pilots, tx_pilots, snr_lin, method='lmmse'):
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
    
    tx_array = UniformLinearArray(num_antennas=num_tx)
    rx_array = UniformLinearArray(num_antennas=num_rx)
    
    true_aoas = torch.tensor([[0.3, 0.0], [-0.5, 0.0], [0.1, 0.0]])
    mpc = MultiPathChannelConfig.random_generate(num_paths=3, max_delay=1e-7, max_doppler=30) 
    mpc.aoas = true_aoas
    
    video_enc = VideoEncoder()
    channel = ISACChannel(tx_array, rx_array, ofdm_cfg, mpc)
    receiver = MIMOReceivers(num_tx)
    sensing = SensingProcessor(rx_array)

    # Payload
    orig_frames = torch.zeros((1, 32, 32), dtype=torch.uint8)
    orig_frames[:, 8:24, 8:24] = 200 
    
    encoded_bits, header_len = video_enc.encode(orig_frames)
    
    bits_per_sym = num_tx * num_data_carrs * 2
    num_data_syms = int(np.ceil(len(encoded_bits) / bits_per_sym))
    
    pad_len = num_data_syms * bits_per_sym - len(encoded_bits)
    if pad_len > 0:
        tx_bits = torch.cat([encoded_bits, torch.zeros(pad_len, dtype=torch.int64)])
    else:
        tx_bits = encoded_bits
        
    data_bits_reshape = tx_bits.reshape(num_tx, num_data_carrs, num_data_syms, 2)
    data_symbols = qpsk_mod(data_bits_reshape).unsqueeze(0) 
    
    snrs = [0, 5, 10, 15, 20, 25]
    pilot_ratios = [0.1, 0.2, 0.3, 0.5]
    
    pareto_results = []
    ber_fec_results = {'uncoded': [], 'fec': []}
    
    best_comm_recon = None
    best_sens_recon = None

    wc = 1.0
    ws = 10.0
    optimal_j = -9999
    optimal_pr = None

    print("Running Pareto Curve Combinatorics (FEC, LMMSE, Root-MUSIC, PSNR vs AoA)...")
    for snr in snrs:
        snr_lin = 10**(snr / 10)
        
        # Uncoded Baseline
        uncoded_bits = torch.randint(0, 2, (len(encoded_bits),), dtype=torch.int64)
        u_pad = num_data_syms * bits_per_sym - len(uncoded_bits)
        u_tx = torch.cat([uncoded_bits, torch.zeros(u_pad, dtype=torch.int64)])
        u_dsyms = qpsk_mod(u_tx.reshape(num_tx, num_data_carrs, num_data_syms, 2)).unsqueeze(0)
        
        def run_chain(d_syms, pr_val, eq_meth):
            Np = max(num_tx, int(np.ceil(num_data_syms * pr_val / (1 - pr_val))))
            p_bits = torch.randint(0, 2, (1, num_tx, num_data_carrs, Np, 2), dtype=torch.int64)
            p_syms = qpsk_mod(p_bits)
            t_grid = torch.cat([p_syms, d_syms], dim=-1)
            
            # Strict Power Normalization: adjust power to maintain constant total energy 
            total_syms = Np + num_data_syms
            t_grid = t_grid / np.sqrt(total_syms)
            
            ofdm_t_grid = ofdm_cfg.get_resource_grid(t_grid)
            
            # Application of Sensing-Centric Transmit Beamforming W
            target_aoa = true_aoas[0, 0].item()
            W = calculate_mrt_matrix(num_tx, target_aoa).to(t_grid.device)
            
            rx_raw, _ = channel.transmit(ofdm_t_grid, snr, W=W)
            
            # Restore unit symbol power for correct LMMSE/Equalization noise baseline
            rx_raw = rx_raw * np.sqrt(total_syms)
            
            rx_d = ofdm_cfg.get_data_grid(rx_raw)[0]
            rx_p = rx_d[:, :, :Np]
            
            # LMMSE Receiver Action
            H_est = receiver.estimate_channel(rx_p, p_syms[0], snr_lin, method='lmmse')
            rx_data_payload = rx_d[:, :, Np:]
            x_hat = receiver.equalize(rx_data_payload, H_est, snr_lin, method=eq_meth)
            
            # Sensing Action
            rx_sens_p = rx_raw[0, :, :, :Np].flatten(start_dim=1)
            try:
                order = estimate_subspace_order(rx_sens_p.unsqueeze(0), method='mdl')[0].item()
                est_aoas = sensing.estimate_aoa_root_music(rx_sens_p, min(order, 3))
                aoa_rmse = sensing.calculate_rmse(true_aoas, est_aoas)
            except Exception:
                aoa_rmse = np.pi
            
            det_bits = qpsk_demod(x_hat).flatten()
            return det_bits, aoa_rmse

        # Uncoded
        u_rx_bits, _ = run_chain(u_dsyms, 0.2, 'mmse')
        u_ber = (u_rx_bits[:len(uncoded_bits)] != uncoded_bits).float().mean().item()
        
        # Coded (FEC) Evaluate once for BER curve tracking
        c_rx_bits, _ = run_chain(data_symbols, 0.2, 'mmse')
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
        
        ber_fec_results['uncoded'].append(u_ber)
        ber_fec_results['fec'].append(c_ber)
        
        # Over Pilot Ratios Sweep mapping Tradeoffs
        for pr in pilot_ratios:
            c_rx_bits, aoa_rmse = run_chain(data_symbols, pr, 'mmse')
            c_rx_trimmed = c_rx_bits[:len(encoded_bits)]
            recon = video_enc.decode(c_rx_trimmed, header_len, 1, (32, 32))
            psnr = calculate_psnr(orig_frames, recon)
            
            rmse_val = max(aoa_rmse, 1e-4) 
            J = wc * psnr - ws * np.log10(rmse_val)
            
            pareto_results.append({'snr': snr, 'pr': pr, 'psnr': psnr, 'aoa': aoa_rmse})
            print(f"SNR:{snr:2d} | PR:{pr:.1f} => PSNR:{psnr:6.1f}dB | AoA RMSE:{aoa_rmse:.4f}rad | J:{J:5.1f}")
            
            if J > optimal_j:
                optimal_j = J
                optimal_pr = pr
                
            if snr == 15 and pr == 0.1: 
                best_comm_recon = recon[0]
            if snr == 15 and pr == 0.5: 
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
