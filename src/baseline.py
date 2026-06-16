""" Non-Adaptive ECG Transmission System """
import os
import sys
import struct
import zlib
import logging
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt
from pyldpc import make_ldpc, encode, decode, get_message
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
DATA_DIR = r"mitdb\mit-bih-arrhythmia-database-1.0.0"
try:
    import wfdb
    HAS_WFDB = True
except Exception:
    HAS_WFDB = False
    logging.info("wfdb not available!")
SEED = 42
np.random.seed(SEED)
FIXED_LDPC_CODE = None


def initialize_fixed_ldpc():
    global FIXED_LDPC_CODE
    n, dv, dc = 252, 3, 6  # Rate 0.50
    H, G = make_ldpc(n, dv, dc, systematic=True, sparse=True)
    k = G.shape[1]
    rate = k / n
    FIXED_LDPC_CODE = {"n": n, "k": k, "H": H, "G": G, "rate": rate}


initialize_fixed_ldpc()
PILOT_LEN = 64
PILOT_SEQ = (np.random.RandomState(12345).choice([-1.0, 1.0], size=PILOT_LEN)).astype(np.float32)
DEFAULT_FS = 360


def load_mitbih_data(record_names, lead=0, segment_size=512, pn_dir=None):
    if pn_dir is None:
        pn_dir = DATA_DIR
    if record_names is None:
        record_names = ['100', '101', '200', '201', '207']
    if not HAS_WFDB:
        raise RuntimeError("WFDB not available")
    all_segments, all_labels = [], []
    fs = DEFAULT_FS
    annotation_map = {
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
        'A': 1, 'a': 1, 'J': 1, 'S': 1, 'V': 1, 'E': 1, '/': 1, 'f': 1, 'Q': 1
    }
    for rec in record_names:
        try:
            record = wfdb.rdrecord(os.path.join(pn_dir, rec), channels=[lead])
            ann = wfdb.rdann(os.path.join(pn_dir, rec), 'atr')
            ecg = record.p_signal.flatten()
            fs = int(record.fs)
            half = segment_size // 2
            for loc, sym in zip(ann.sample, ann.symbol):
                if sym not in annotation_map:
                    continue
                start = max(0, loc - half)
                end = start + segment_size
                if end <= len(ecg):
                    all_segments.append(ecg[start:end].copy())
                    all_labels.append(annotation_map[sym])
        except Exception as e:
            logging.warning(f"Failed to load record {rec}: {e}")
    if len(all_segments) == 0:
        raise RuntimeError("No segments found in mitdb")
    return np.array(all_segments), np.array(all_labels), fs


# ----- PREPROCESSING -----
def preprocess_ecg(segment, fs, lowcut=0.5, highcut=40.0, notch_freq=50.0):
    nyq = 0.5 * fs
    b_hp, a_hp = butter(2, lowcut / nyq, btype='high')
    s = filtfilt(b_hp, a_hp, segment)
    low = (notch_freq - 1.0) / nyq
    high = (notch_freq + 1.0) / nyq
    if low > 0 and high < 1:
        b_nt, a_nt = butter(2, [low, high], btype='bandstop')
        s = filtfilt(b_nt, a_nt, s)
    m, sd = np.mean(s), np.std(s)
    if sd > 1e-7:
        return (s - m) / sd
    else:
        return s - m


# ----- FIXED COMPRESSION -----
FIXED_THRESHOLD = 0.12
FIXED_WAVELET = 'db4'
FIXED_LEVEL = 4


def dead_zone_quantizer(coeffs, T):
    return [np.where(np.abs(c) >= T, np.round(c / T).astype(np.int16), np.int16(0)) for c in coeffs]


def inverse_dead_zone_quantizer(quantized_coeffs, T):
    return [c.astype(np.float64) * T for c in quantized_coeffs]


def compress_signal(signal):
    coeffs = pywt.wavedec(signal, FIXED_WAVELET, level=FIXED_LEVEL, mode='periodization')
    q = dead_zone_quantizer(coeffs, FIXED_THRESHOLD)
    shapes = [len(c) for c in q]
    return q, shapes


def decompress_signal(qcoeffs, target_len):
    coeffs = inverse_dead_zone_quantizer(qcoeffs, FIXED_THRESHOLD)
    rec = pywt.waverec(coeffs, FIXED_WAVELET, mode='periodization')
    return rec[:target_len]


# ----- PACKETIZATION -----
def create_packet(quantized_coeffs, shapes):
    flat = np.concatenate([c.flatten() for c in quantized_coeffs]).astype(np.int16)
    total_coeffs = len(flat)
    nonzero_indices = np.nonzero(flat)[0]
    nonzero_values = flat[nonzero_indices]
    num_shapes = len(shapes)
    if len(nonzero_indices) / total_coeffs > 0.5:
        payload = flat.tobytes()  # Dense
        num_nonzeros = total_coeffs
    else:
        indices_bytes = nonzero_indices.astype(np.uint16).tobytes()
        values_bytes = nonzero_values.astype(np.int16).tobytes()
        payload = indices_bytes + values_bytes  # Sparse
        num_nonzeros = len(nonzero_indices)
    header = struct.pack('I', num_shapes)
    header += struct.pack(f'{num_shapes}I', *shapes)
    header += struct.pack('I', num_nonzeros)
    header += struct.pack('I', len(payload))
    full = header + payload
    crc = zlib.crc32(full) & 0xFFFFFFFF
    packet = struct.pack('I', crc) + full
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    return bits


def extract_packet(bitstream):
    b = np.array(bitstream, dtype=np.uint8)
    pad = (-len(b)) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    data = np.packbits(b).tobytes()
    if len(data) < 20:
        return None, False, "Packet too short!"
    crc_expected = struct.unpack('I', data[:4])[0]
    num_shapes = struct.unpack('I', data[4:8])[0]
    if num_shapes <= 0 or num_shapes > 100:
        return None, False, "Invalid num_shapes!"
    header_size = 4 + 4 + num_shapes * 4 + 4 + 4
    if len(data) < header_size:
        return None, False, "Header incomplete!"
    shapes = list(struct.unpack(f'{num_shapes}I', data[8:8 + 4 * num_shapes]))
    num_nonzeros = struct.unpack('I', data[8 + 4 * num_shapes:12 + 4 * num_shapes])[0]
    payload_len = struct.unpack('I', data[12 + 4 * num_shapes:header_size])[0]
    payload_end = header_size + payload_len
    if payload_end > len(data):
        return None, False, "Payload incomplete!"
    crc_actual = zlib.crc32(data[4:payload_end]) & 0xFFFFFFFF
    if crc_expected != crc_actual:
        return None, False, "CRC mismatch!"
    payload = data[header_size:payload_end]
    total_coeffs = sum(shapes)
    if num_nonzeros == total_coeffs or num_nonzeros == payload_len:
        flat = np.frombuffer(payload, dtype=np.int16)
    else:
        half_len = len(payload) // 2
        indices = np.frombuffer(payload[:half_len], dtype=np.uint16)
        values = np.frombuffer(payload[half_len:], dtype=np.int16)
        flat = np.zeros(total_coeffs, dtype=np.int16)
        flat[indices] = values
    coeffs, idx = [], 0
    for s in shapes:
        coeffs.append(flat[idx:idx + s])
        idx += s
    return coeffs, True, "OK"


# ----- LDPC ENCODE/DECODE -----
def ldpc_encode(bits):
    code = FIXED_LDPC_CODE
    k, n = code["k"], code["n"]
    pad = (-len(bits)) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    blocks = bits.reshape(-1, k)
    encoded = []
    for blk in blocks:
        try:
            x = encode(code["G"], blk.astype(int), snr=1000)
            cw = (x < 0).astype(np.uint8)
            encoded.extend(cw)
        except Exception as e:
            logging.warning(f"LDPC encode exception: {e}")
            cw = np.zeros(n, dtype=np.uint8)
            cw[:k] = blk
            encoded.extend(cw)
    return np.array(encoded, dtype=np.uint8), pad


def ldpc_decode(received_symbols, snr_db, pad):
    code = FIXED_LDPC_CODE
    rcv = np.array(received_symbols, dtype=np.float64)
    n, k = int(code["n"]), int(code["k"])
    extra = (-len(rcv)) % n
    if extra > 0:
        rcv = np.concatenate([rcv, np.zeros(extra, dtype=rcv.dtype)])
    blocks = rcv.reshape(-1, n)
    decoded_bits = []
    failed = 0
    for blk in blocks:
        try:
            maxiter = 200
            if snr_db is not None and snr_db < 10:
                maxiter = 300
            d = decode(code["H"], blk, snr_db, maxiter=maxiter)
            msg = get_message(code["G"], d)
            decoded_bits.extend([int(x) for x in msg.tolist()])
        except Exception:
            hard_bits = (blk < 0).astype(np.uint8)
            decoded_bits.extend(hard_bits[:k].tolist())
            failed += 1
    decoded = np.array(decoded_bits, dtype=np.uint8)
    if pad and pad > 0:
        if pad <= len(decoded):
            decoded = decoded[:-pad]
    return decoded, failed


# ----- PHYSICAL LAYER -----
def bpsk_modulate(bits):
    return np.where(bits == 0, 1.0, -1.0).astype(np.float32)


def insert_pilots(symbols, pilot_spacing=128):
    out = []
    idx = 0
    while idx < len(symbols):
        out.extend(PILOT_SEQ.tolist())
        chunk = symbols[idx:idx + pilot_spacing]
        out.extend(chunk.tolist())
        idx += pilot_spacing
    return np.array(out, dtype=np.complex64)


def rician_awgn_channel(symbols, snr_db, K_db=7.0):
    K = 10 ** (K_db / 10.0)
    LOS_amp = np.sqrt(K / (K + 1.0))
    SCAT_amp = np.sqrt(1.0 / (K + 1.0))
    N = len(symbols)
    phase_drift = 0.001 * np.cumsum(np.random.randn(N))
    los = LOS_amp * np.exp(1j * phase_drift)
    block_size = 256
    num_blocks = (N + block_size - 1) // block_size
    h_scatter_blocks = []
    for _ in range(num_blocks):
        fade = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2.0)
        h_scatter_blocks.extend([fade] * block_size)
    h_scatter = SCAT_amp * np.array(h_scatter_blocks[:N])
    h = los + h_scatter
    faded = h * symbols
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = np.mean(np.abs(faded) ** 2) / max(snr_lin, 1e-9)
    noise = np.sqrt(noise_power / 2.0) * (np.random.randn(N) + 1j * np.random.randn(N))
    rx = faded + noise
    return rx, h


# ----- TRANSMITTER -----
def transmitter(signal):
    quantized, shapes = compress_signal(signal)
    packet_bits = create_packet(quantized, shapes).astype(np.uint8)
    coded_bits, pad_bits = ldpc_encode(packet_bits)
    coded_bits = np.array(coded_bits, dtype=np.uint8)
    n = FIXED_LDPC_CODE["n"]
    interleaver_extra = (-len(coded_bits)) % n
    if interleaver_extra > 0:
        coded_bits = np.concatenate([coded_bits, np.zeros(interleaver_extra, dtype=np.uint8)])
    try:
        coded_bits = coded_bits.reshape(-1, n).T.flatten()
    except Exception:
        pass
    coded_bits = np.array(coded_bits, dtype=np.uint8)
    original_bits = signal.size * 16
    packet_bits_len = int(len(packet_bits))
    coded_bits_len = int(len(coded_bits))
    CR = original_bits / packet_bits_len if packet_bits_len > 0 else np.nan
    return {
        "bits": coded_bits,
        "pad": pad_bits,
        "interleaver_pad": interleaver_extra,
        "shapes": shapes,
        "packet_bits_len": packet_bits_len,
        "coded_bits_len": coded_bits_len,
        "original_bits": original_bits,
        "CR": CR
    }


# ----- RECEIVER -----
def receiver(received, tx_info, snr_db, signal_len=512):
    crc_failed = False
    pilot_len = len(PILOT_SEQ)
    pilot_spacing = 128
    data_syms = []
    idx = 0
    L = len(received)
    while idx < L:
        if idx + pilot_len <= L:
            idx += pilot_len
        if idx + pilot_spacing <= L:
            data_syms.extend(received[idx:idx + pilot_spacing])
            idx += pilot_spacing
        else:
            data_syms.extend(received[idx:])
            break
    data_symbols = np.array(data_syms)
    equalized_real = np.real(data_symbols)
    n = int(FIXED_LDPC_CODE["n"])
    interleaver_pad = int(tx_info.get('interleaver_pad', 0))
    num_cols = len(equalized_real) // n
    if num_cols > 0:
        trimmed = equalized_real[:num_cols * n]
        try:
            deintl = trimmed.reshape(n, num_cols).T.flatten()
            if interleaver_pad > 0 and interleaver_pad < len(deintl):
                deintl = deintl[:-interleaver_pad]
            rest = equalized_real[num_cols * n:]
            equalized_real = np.concatenate([deintl, rest]).astype(np.float64)
        except Exception:
            pass
    soft_symbols = equalized_real
    soft_symbols = np.clip(soft_symbols, -50.0, 50.0)
    try:
        decoded_bits, ldpc_fails = ldpc_decode(soft_symbols, snr_db, tx_info.get('pad', 0))
    except Exception as e:
        return np.zeros(signal_len), False, f"ldpc_fail: {e}", 1, True
    try:
        coeffs, crc_ok, msg = extract_packet(decoded_bits)
        if not crc_ok:
            crc_failed = True
    except Exception as e:
        coeffs, crc_ok, msg = None, False, f"packet_fail: {e}"
        crc_failed = True
    if crc_ok and coeffs is not None:
        try:
            reconstructed = decompress_signal(coeffs, signal_len)
            return reconstructed, True, msg, ldpc_fails, crc_failed
        except Exception as e:
            return np.zeros(signal_len), False, f"decompress_fail: {e}", ldpc_fails, True
    else:
        return np.zeros(signal_len), False, msg, 0, True


# ----- DIAGNOSTICS -----
def run_diagnostics(segments, labels, snr_values=(8, 10, 12, 15, 20, 25)):
    K_db = 7.0
    pilot_spacing = 128
    print("\nDiagnostic Configuration:")
    print(f"  Testing {len(snr_values)} SNR points")
    results = []
    for snr in snr_values:
        print(f"\n  Testing SNR={snr}dB...")
        success = 0
        prds = []
        total_tested = 0
        failed_ldpc = 0
        failed_crc = 0
        for i in range(len(segments)):
            signal = preprocess_ecg(segments[i], DEFAULT_FS)
            try:
                tx = transmitter(signal)
                tx_syms = bpsk_modulate(tx['bits'])
                tx_with_pilots = insert_pilots(tx_syms, pilot_spacing=pilot_spacing)
                rx, _ = rician_awgn_channel(tx_with_pilots, snr_db=snr, K_db=K_db)
                rec, ok, msg, fails, crc_fail = receiver(rx, tx, snr, signal_len=512)
                if ok:
                    prd = compute_prd(signal, rec)
                    prds.append(prd)
                    success += 1
                else:
                    if fails > 0:
                        failed_ldpc += 1
                    if crc_fail:
                        failed_crc += 1
                total_tested += 1
            except Exception:
                total_tested += 1
        succ_rate = success / total_tested if total_tested > 0 else 0
        mean_prd = np.nanmean(prds) if len(prds) > 0 else np.nan
        results.append({
            'snr': snr,
            'success_rate': succ_rate,
            'mean_prd': mean_prd,
            'total_tested': total_tested,
            'successful': success,
            'failed': total_tested - success,
            'ldpc_failures': failed_ldpc,
            'crc_failures': failed_crc
        })
        logging.info(
            f"SNR {snr}dB -> "
            f"success {succ_rate * 100:.1f}% "
            f"({success}/{total_tested}) "
            f"mean PRD {mean_prd:.3f}%"
        )
    snrs = [r['snr'] for r in results]
    succ = [r['success_rate'] for r in results]
    prd = [r['mean_prd'] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(snrs, np.array(succ) * 100, '-o', linewidth=2, markersize=8, color='red', label='Baseline (Fixed)')
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Baseline System: Success vs SNR\n(Balanced: 50 Normal + 50 Arrhythmia)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    ax1.legend()
    ax2.plot(snrs, prd, '-o', linewidth=2, markersize=8, color='orange', label='Baseline PRD')
    ax2.axhline(y=9, color='r', linestyle='--', linewidth=2, label='Clinical Threshold (9%)')
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Mean PRD (%)', fontsize=12)
    ax2.set_title('Baseline System: Quality (Mean PRD) vs. Channel SNR\n(Fixed Compression T=0.12)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 15])
    plt.tight_layout()
    plt.savefig('baseline_system_results.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  Diagnostic Summary:")
    print("  " + "=" * 80)
    print(f"  {'SNR':>6} {'Success':>10} {'Failed':>8} {'LDPC':>8} {'CRC':>8} {'Mean PRD':>10}")
    print("  " + "-" * 80)
    for r in results:
        print(f"  {r['snr']:>4}dB {r['successful']:>4}/{r['total_tested']:<3} "
              f"{r['failed']:>6} {r['ldpc_failures']:>8} {r['crc_failures']:>8} "
              f"{r['mean_prd']:>9.2f}%")
    print("  " + "=" * 80)
    return results


# ----- METRICS -----
def compute_prd(o, r):
    L = min(len(o), len(r))
    return 100.0 * np.sqrt(np.sum((o[:L] - r[:L]) ** 2) / (np.sum(o[:L] ** 2) + 1e-12))


def compute_mse(original, reconstructed):
    L = min(len(original), len(reconstructed))
    return np.mean((original[:L] - reconstructed[:L]) ** 2)


# ----- MAIN -----
def main():
    print("\n" + "=" * 70)
    print("BASELINE (NON-ADAPTIVE) ECG TRANSMISSION SYSTEM")
    print("=" * 70)
    print("\n[STEP 1] Loading ECG Data:")
    try:
        segments, labels, fs = load_mitbih_data(None)
        print(f" Loaded {len(segments)} segments from MIT-BIH database")
        print(f"  Normal: {np.sum(labels == 0)} segments")
        print(f"  Arrhythmia: {np.sum(labels == 1)} segments")
    except Exception as e:
        print(f"ERROR: Failed to load MIT-BIH data: {e}")
        sys.exit(1)
    print("\n[STEP 2] Creating Balanced Test Set:")
    normal_idx = np.where(labels == 0)[0]
    arrhythmia_idx = np.where(labels == 1)[0]
    np.random.seed(42)
    selected_normal = np.random.choice(normal_idx, size=50, replace=False)
    selected_arrhythmia = np.random.choice(arrhythmia_idx, size=50, replace=False)
    test_indices = np.concatenate([selected_normal, selected_arrhythmia])
    np.random.shuffle(test_indices)
    test_segments = segments[test_indices]
    test_labels = labels[test_indices]
    print(f" Created balanced test set: 100 segments")
    print(f"  Normal: {np.sum(test_labels == 0)} segments")
    print(f"  Arrhythmia: {np.sum(test_labels == 1)} segments")
    print("\n[STEP 3] Preprocessing ECG Signals:")
    X_preprocessed = []
    for i in range(len(test_segments)):
        preprocessed = preprocess_ecg(test_segments[i], fs)
        X_preprocessed.append(preprocessed)
    X_preprocessed = np.array(X_preprocessed)
    print(f" Preprocessed {len(test_segments)} segments")
    print("\n[STEP 4] Running Diagnostics:")
    diagnostic_results = run_diagnostics(
        test_segments,
        test_labels,
        snr_values=(2, 4, 6, 8, 10, 12, 15, 20, 25),
    )
    print("\n[STEP 5] Computing Energy Metrics:")
    energy_samples = []
    for i in range(20):
        idx = np.random.randint(0, len(test_segments))
        signal = preprocess_ecg(test_segments[idx], DEFAULT_FS)
        tx = transmitter(signal)
        tx_syms = bpsk_modulate(tx['bits'])
        tx_with_pilots = insert_pilots(tx_syms, pilot_spacing=128)
        energy_samples.append(len(tx_with_pilots))
    avg_baseline_symbols = np.mean(energy_samples)
    avg_adaptive_symbols = 6208
    print(f"  Baseline System (avg over 20 samples):")
    print(f"    Transmitted symbols: {avg_baseline_symbols:.0f}")
    print(f"    Original bits: 8192")
    print(f"    Compression + overhead: {avg_baseline_symbols / 8192:.3f}×")
    print(f"\n  Energy Comparison:")
    print(f"    Baseline transmits: {avg_baseline_symbols:.0f} symbols")
    print(f"    Adaptive transmits: {avg_adaptive_symbols:.0f} symbols")
    print(f"    Adaptive saves: {(avg_baseline_symbols - avg_adaptive_symbols) / avg_baseline_symbols * 100:.1f}%")
    print("\n✓ BASELINE SIMULATION COMPLETE\n")


if __name__ == "__main__":
    main()