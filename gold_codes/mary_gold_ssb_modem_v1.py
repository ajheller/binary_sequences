"""M-ary Gold SSB modem V1.0; over-the-air frame format remains version 3."""

from dataclasses import dataclass
import zlib
from time import perf_counter

import numpy as np
from scipy import signal
from scipy.io import wavfile

# ==============================================================================
# 1. EXPLICIT SYSTEM & SIMULATION PARAMETERS
# ==============================================================================
SOFTWARE_VERSION = "1.0"

FS = 8000  # Soundcard sample rate (Hz)
F_SUB = 1500.0  # Audio subcarrier center frequency (Hz)
CHIP_RATE = 1600  # Chips per second (1600 * 1.25 = 2000 Hz BW -> 500..2500 Hz)
SPS = FS // CHIP_RATE  # Samples per chip = 5

# Gold Code Length Configuration:
#   7 -> 127 chips (~79.4 ms/sym, ~50.4 bps, +21.0 dB processing gain)
#   Degree 8 is unsupported: preferred Gold pairs require n not divisible by 4.
#   9 -> 511 chips (~319.4 ms/sym, ~12.5 bps, +27.1 dB processing gain)
LFSR_DEGREE = 9
CODE_LEN = (1 << LFSR_DEGREE) - 1

M_ARY = 16  # 16 data Gold codes -> 4 bits per symbol
BITS_PER_SYM = int(np.log2(M_ARY))
ROLLOFF = 0.25  # RRC filter excess bandwidth factor
RRC_SPAN = 8  # RRC filter span in chips
PREAMBLE_LEN = 4  # Number of repeated preamble symbols for packet detection

# Pilot controls. Waveform settings travel in the common header; the receiver
# does not need matching local pilot settings. Tracking is receiver-local.
PILOT_MODE = "off"  # "off", "periodic", or "continuous"
PILOT_POWER_FRACTION = 0.10  # Continuous: nominal component power, 0.01..0.50
PILOT_INTERVAL = 8  # Periodic: data symbols between pilot bursts (1..255)
PILOT_BURST_LENGTH = 1  # Periodic: pilot symbols per burst (1..16)
PILOT_TRACKING = True  # Frequency/timing tracking; no equalization yet

# Crest-Factor Reduction (CFR) Parameters:
ENABLE_CFR = True
TARGET_PAPR_DB = 5.5  # Target Peak-to-Average Power Ratio (dB)
CFR_ITERATIONS = 4

# Monte Carlo BER vs. SNR Sweep Parameters:
RUN_SNR_SWEEP = True
SNR_SWEEP_DB = [-6.0, -9.0, -12.0, -14.0, -16.0, -18.0]  # Referenced to 2.4 kHz SSB BW
TRIALS_PER_SNR = 10
VFO_OFFSET_HZ = 38.5  # Simulated SSB VFO tuning offset (Hz)
MAX_DOPPLER_HZ = 100.0  # Receiver frequency search range (+/- Hz)

# ==============================================================================
# 2. GOLD CODE GENERATOR (Validated Preferred Pairs for Degrees 7 and 9)
# ==============================================================================
PREFERRED_TAPS = {
    7: ([7, 3], [7, 3, 2, 1]),
    9: ([9, 4], [9, 6, 4, 3]),
}


def generate_m_sequence(taps, reg_len):
    """Generates a maximal-length sequence (+/-1) from feedback taps."""
    state = np.ones(reg_len, dtype=int)
    length = (1 << reg_len) - 1
    seq = np.zeros(length, dtype=int)
    for i in range(length):
        seq[i] = state[-1]
        feedback = 0
        for t in taps:
            feedback ^= state[t - 1]
        state[1:] = state[:-1]
        state[0] = feedback
    return 1 - 2 * seq  # Map {0, 1} -> {+1, -1}


def generate_gold_codes(lfsr_degree=LFSR_DEGREE, num_data_codes=16):
    """Returns m1 preamble and data codes; both m-sequences are reserved."""
    if lfsr_degree not in PREFERRED_TAPS:
        raise ValueError(f"Supported LFSR degrees: {list(PREFERRED_TAPS.keys())}")
    if not 1 <= num_data_codes <= (1 << lfsr_degree) - 1:
        raise ValueError("Data code count exceeds the distinct Gold family size.")
    taps1, taps2 = PREFERRED_TAPS[lfsr_degree]
    m1 = generate_m_sequence(taps1, lfsr_degree)
    m2 = generate_m_sequence(taps2, lfsr_degree)

    # Check maximal periods through periodic autocorrelation, and verify the
    # preferred pair under this implementation's actual shift-register convention.
    for seq in (m1, m2):
        ac = np.rint(np.fft.ifft(np.abs(np.fft.fft(seq)) ** 2).real).astype(int)
        if ac[0] != len(seq) or not np.all(ac[1:] == -1):
            raise ValueError("Configured taps do not generate an m-sequence.")
    cc = np.rint(np.fft.ifft(np.fft.fft(m1) * np.conj(np.fft.fft(m2))).real).astype(int)
    bound = 1 + (1 << ((lfsr_degree + 1) // 2))
    if set(cc) != {-bound, -1, bound - 2}:
        raise ValueError("Configured sequences are not a preferred Gold pair.")

    # Replace the former m2 data slot with the next unused product code;
    # preserve the other data-symbol assignments. m2 is payload-pilot only.
    codes = [m1, m1 * np.roll(m2, num_data_codes - 1)]
    for shift in range(num_data_codes - 1):
        gold_seq = m1 * np.roll(m2, shift)
        codes.append(gold_seq)
    codes = np.array(codes, dtype=float)
    if len(np.unique(codes, axis=0)) != len(codes):
        raise ValueError("Duplicate Gold codes.")
    return codes[0], codes[1:]  # preamble_code, data_codebook


def payload_pilot_code(preamble_code):
    """Return the reserved m2 paired with a supported canonical m1 preamble."""
    for degree, (taps1, taps2) in PREFERRED_TAPS.items():
        if len(preamble_code) == (1 << degree) - 1:
            if not np.array_equal(preamble_code, generate_m_sequence(taps1, degree)):
                raise ValueError(
                    "Expected the canonical m1 preamble for this code length."
                )
            return generate_m_sequence(taps2, degree).astype(float)
    raise ValueError("Unsupported preamble length for payload pilot.")


# Frame v3, high nibble first. Header: MG, version=3, length, mode,
# power percent, interval, burst length, then big-endian CRC-32 of those 8 bytes.
# Payload trailer: big-endian CRC-32 of the complete header + payload.
# m1 is preamble-only; header uses plain data codes. m2 pilots apply only
# to payload + trailer. No pilot is transmitted between packets.
FRAME_MARKER = b"MG\x03"
CRC_BYTES = 4
HEADER_BODY_BYTES = 8
HEADER_BYTES = HEADER_BODY_BYTES + CRC_BYTES
MIN_FRAME_BYTES = HEADER_BYTES + CRC_BYTES
MIN_PREAMBLE_SCORE = 0.06
MIN_REPEAT_SCORE = 0.02
PILOT_MODES = ("off", "periodic", "continuous")


@dataclass(frozen=True)
class PilotConfig:
    mode: str = "off"
    power_fraction: float = 0.10
    interval: int = 8
    burst_length: int = 1

    def __post_init__(self):
        if self.mode not in PILOT_MODES:
            raise ValueError("Pilot mode must be off, periodic, or continuous.")
        if (
            not np.isfinite(self.power_fraction)
            or not 0.01 <= self.power_fraction <= 0.50
            or not np.isclose(
                self.power_fraction * 100,
                round(self.power_fraction * 100),
                rtol=0,
                atol=1e-8,
            )
        ):
            raise ValueError("Pilot power must be a whole percent from 1% to 50%.")
        object.__setattr__(
            self, "power_fraction", round(self.power_fraction * 100) / 100
        )
        if not isinstance(self.interval, int) or not 1 <= self.interval <= 255:
            raise ValueError("Pilot interval must be an integer in 1..255.")
        if not isinstance(self.burst_length, int) or not 1 <= self.burst_length <= 16:
            raise ValueError("Pilot burst length must be an integer in 1..16.")

    def wire_bytes(self):
        # Canonical zeroes for settings unused by a mode.
        return bytes(
            [
                PILOT_MODES.index(self.mode),
                round(self.power_fraction * 100) if self.mode == "continuous" else 0,
                self.interval if self.mode == "periodic" else 0,
                self.burst_length if self.mode == "periodic" else 0,
            ]
        )


def configured_pilot():
    return PilotConfig(
        PILOT_MODE, PILOT_POWER_FRACTION, PILOT_INTERVAL, PILOT_BURST_LENGTH
    )


@dataclass(frozen=True)
class DecodeResult:
    payload: bytes | None
    status: str
    sync_idx: int = -1
    vfo_hz: float = 0.0
    preamble_score: float = 0.0
    raw_payload: bytes | None = None
    pilot_config: PilotConfig | None = None
    pilot_observations: int = 0
    tracking_updates: int = 0
    final_vfo_hz: float = 0.0
    timing_offset_samples: int = 0

    @property
    def ok(self):
        return self.status == "ok"


@dataclass(frozen=True)
class TransmitResult:
    audio: np.ndarray
    papr_before_db: float
    papr_after_db: float
    pilot_config: PilotConfig
    pilot_symbols: int
    body_data_symbols: int
    composite_chip_power: float  # Measured before continuous-symbol normalization
    active_audio_power: float  # Measured after CFR and final file scaling


def encode_frame(payload, pilot_config=None):
    config = configured_pilot() if pilot_config is None else pilot_config
    payload = bytes(payload)
    if len(payload) > 255:
        raise ValueError("Max payload length is 255 bytes.")
    header_body = FRAME_MARKER + bytes([len(payload)]) + config.wire_bytes()
    header = header_body + zlib.crc32(header_body).to_bytes(CRC_BYTES, "big")
    body = header + payload
    return body + zlib.crc32(body).to_bytes(CRC_BYTES, "big")


def parse_header(header):
    """Return (length, config), or None for a corrupt/unsupported header."""
    if len(header) != HEADER_BYTES or header[:3] != FRAME_MARKER:
        return None
    if zlib.crc32(header[:HEADER_BODY_BYTES]) != int.from_bytes(
        header[HEADER_BODY_BYTES:], "big"
    ):
        return None
    length, mode, power, interval, burst = header[3:8]
    try:
        if mode == 0 and (power, interval, burst) == (0, 0, 0):
            config = PilotConfig()
        elif mode == 1 and power == 0:
            config = PilotConfig("periodic", interval=interval, burst_length=burst)
        elif mode == 2 and (interval, burst) == (0, 0):
            config = PilotConfig("continuous", power_fraction=power / 100)
        else:
            return None
    except ValueError:
        return None
    return length, config


def validate_frame(frame):
    if len(frame) < HEADER_BYTES:
        return None, "truncated"
    header = parse_header(frame[:HEADER_BYTES])
    if header is None:
        return None, "invalid_header"
    expected = MIN_FRAME_BYTES + header[0]
    if len(frame) != expected:
        return None, "truncated" if len(frame) < expected else "invalid_length"
    if zlib.crc32(frame[:-CRC_BYTES]) != int.from_bytes(frame[-CRC_BYTES:], "big"):
        return None, "crc_failed"
    return frame[HEADER_BYTES:-CRC_BYTES], "ok"


def body_schedule(data_symbols, config):
    """Data index or None (pilot); no trailing periodic burst."""
    schedule = []
    for i in range(data_symbols):
        if config.mode == "periodic" and i and i % config.interval == 0:
            schedule.extend([None] * config.burst_length)
        schedule.append(i)
    return schedule


def body_templates(preamble_code, data_codebook, config):
    if config.mode != "continuous":
        return np.asarray(data_codebook), np.ones(len(data_codebook))
    mixed = np.sqrt(1 - config.power_fraction) * data_codebook + np.sqrt(
        config.power_fraction
    ) * payload_pilot_code(preamble_code)
    power = np.mean(mixed**2, axis=1)
    # Cross terms are real: normalize each composite symbol, and use the same
    # templates at RX. No extra total chip power is granted to continuous mode.
    return mixed / np.sqrt(power[:, None]), power


def pilot_summary(config, code_len, payload_length):
    count = 2 * (payload_length + CRC_BYTES)
    schedule = body_schedule(count, config)
    pilots = schedule.count(None)
    if config.mode == "periodic":
        spacing_ms = (
            (config.interval + config.burst_length) * code_len / CHIP_RATE * 1000
        )
        return (
            f"Periodic: {config.burst_length} pilot(s) / {config.interval} data symbols | "
            f"Burst-start spacing: {spacing_ms:.1f} ms | "
            f"Body pilot airtime: {100 * pilots / len(schedule):.1f}% ({pilots} symbols)"
        )
    if config.mode == "continuous":
        return (
            f"Continuous: nominal pilot/data component power "
            f"{100 * config.power_fraction:.0f}%/{100 * (1-config.power_fraction):.0f}% | "
            f"Observation interval: {1000 * code_len / CHIP_RATE:.1f} ms | No extra symbols"
        )
    return "Off: no payload pilot power or airtime overhead"


# ==============================================================================
# 3. ROOT-RAISED COSINE (RRC) & CREST-FACTOR REDUCTION (CFR)
# ==============================================================================
def rrc_filter(sps, span, alpha):
    """Designs a unit-energy Root-Raised Cosine FIR filter."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)
    t = n / float(sps)
    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif alpha != 0 and np.isclose(abs(ti), 1.0 / (4.0 * alpha)):
            h[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(
                np.pi * ti * (1.0 + alpha)
            )
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den

    return h / np.sqrt(np.sum(h**2))


def apply_cfr(audio_sig, target_papr_db=5.5, iterations=4):
    """Iterative Clip-and-Filter Crest-Factor Reduction (CFR)."""
    sos_bp = signal.butter(6, [350.0, 2650.0], btype="bandpass", fs=FS, output="sos")
    y = audio_sig.copy()
    for _ in range(iterations):
        rms = np.sqrt(np.mean(y**2))
        clip_thresh = rms * (10.0 ** (target_papr_db / 20.0))
        y = np.clip(y, -clip_thresh, clip_thresh)
        y = signal.sosfiltfilt(sos_bp, y)
    papr_final = 10.0 * np.log10(np.max(y**2) / np.mean(y**2))
    return y, papr_final


# ==============================================================================
# 4. TRANSMITTER & SIMULATED SSB CHANNEL
# ==============================================================================
def modulate_packet(
    payload_bytes,
    preamble_code,
    data_codebook,
    rrc_taps,
    *,
    pilot_config=None,
    return_details=False,
):
    """Encode a v3 packet; optional details include measured power diagnostics."""
    if M_ARY != 16 or BITS_PER_SYM != 4:
        raise ValueError("Frame v3 requires 16-ary modulation.")
    if np.shape(data_codebook) != (M_ARY, len(preamble_code)):
        raise ValueError("Expected 16 data codes matching the preamble length.")
    config = configured_pilot() if pilot_config is None else pilot_config
    frame = np.frombuffer(encode_frame(payload_bytes, config), dtype=np.uint8)
    indices = np.column_stack((frame >> 4, frame & 15)).ravel()
    header_symbols = data_codebook[indices[: HEADER_BYTES * 2]]
    body_indices = indices[HEADER_BYTES * 2 :]
    templates, composite_power = body_templates(preamble_code, data_codebook, config)
    schedule = body_schedule(len(body_indices), config)
    pilot_code = payload_pilot_code(preamble_code)
    body_symbols = np.array(
        [pilot_code if i is None else templates[body_indices[i]] for i in schedule]
    )
    all_chips = np.concatenate(
        [
            np.tile(preamble_code, PREAMBLE_LEN),
            header_symbols.ravel(),
            body_symbols.ravel(),
        ]
    )
    upsampled = np.zeros(len(all_chips) * SPS)
    upsampled[::SPS] = all_chips
    baseband = signal.convolve(upsampled, rrc_taps, mode="full")
    t = np.arange(len(baseband)) / float(FS)
    audio_raw = baseband * np.cos(2.0 * np.pi * F_SUB * t)
    papr_before = 10 * np.log10(np.max(audio_raw**2) / np.mean(audio_raw**2))
    if ENABLE_CFR:
        audio_tx, papr_after = apply_cfr(audio_raw, TARGET_PAPR_DB, CFR_ITERATIONS)
    else:
        audio_tx, papr_after = audio_raw, papr_before
    scale = 0.8 / np.max(np.abs(audio_tx))
    silence = np.zeros(int(0.25 * FS))
    audio_final = np.concatenate([silence, audio_tx * scale, silence])
    if return_details:
        return TransmitResult(
            audio_final,
            papr_before,
            papr_after,
            config,
            schedule.count(None),
            len(body_indices),
            float(np.mean(composite_power[body_indices])),
            float(np.mean((audio_tx * scale) ** 2)),
        )
    return audio_final, papr_before, papr_after


def pass_through_ssb_channel(
    audio_tx, snr_db, vfo_offset_hz, extra_delay_samples=0, rng=None
):
    """Simulates SSB bandpass filtering, VFO frequency shift, and AWGN in 2.4 kHz BW."""
    if rng is None:
        rng = np.random.default_rng()

    if extra_delay_samples > 0:
        rx = np.concatenate([np.zeros(extra_delay_samples), audio_tx, np.zeros(200)])
    else:
        rx = audio_tx.copy()

    t = np.arange(len(rx)) / float(FS)
    rx_shifted = np.real(
        signal.hilbert(rx) * np.exp(1j * 2.0 * np.pi * vfo_offset_hz * t)
    )
    sos_ssb = signal.butter(6, [300.0, 2700.0], btype="bandpass", fs=FS, output="sos")
    rx_filtered = signal.sosfilt(sos_ssb, rx_shifted)

    # Measure active packet RMS power (excluding 0.25s leading/trailing silence)
    lead_samples = int(0.25 * FS) + extra_delay_samples
    trail_samples = int(0.25 * FS) + (200 if extra_delay_samples > 0 else 0)
    active_seg = rx_filtered[lead_samples : len(rx_filtered) - trail_samples]
    active_power = np.mean(active_seg**2)

    # Calibrate AWGN to 2,400 Hz reference SSB bandwidth
    noise_std = np.sqrt(
        (active_power / (10.0 ** (snr_db / 10.0))) * (FS / 2.0) / 2400.0
    )
    rx_noisy = rx_filtered + rng.normal(0.0, noise_std, len(rx_filtered))
    return rx_noisy


# ==============================================================================
# 5. NON-COHERENT 2D FFT RECEIVER
# ==============================================================================
class PilotTracker:
    """Confidence-gated, per-symbol frequency/sample-timing tracker.

    Observes m2 without subtracting data. This is a single-path tracking aid,
    not a multipath estimator/equalizer. Weak pilots can produce no updates.
    """

    def __init__(self, frequency, max_doppler_hz, code, config, n_fft):
        self.frequency = frequency
        self.max_doppler_hz = max_doppler_hz
        self.code = code
        self.config = config
        self.n_fft = n_fft
        self.freqs = np.fft.fftfreq(n_fft, d=1 / CHIP_RATE)
        self.timing = 0
        self.updates = 0

    def observe(self, bb, nominal_start):
        bins = np.where(
            (np.abs(self.freqs - self.frequency) <= 12.0)
            & (np.abs(self.freqs) <= self.max_doppler_hz)
        )[0]
        if not len(bins):
            return
        best = None
        n = len(self.code)
        for delta in range(-2, 3):
            offset = self.timing + delta
            start = nominal_start + offset
            if start < 0 or start + n * SPS > len(bb):
                continue
            chips = bb[start : start + n * SPS : SPS]
            energy = n * np.sum(np.abs(chips) ** 2)
            if energy <= 0:
                continue
            power = np.abs(np.fft.fft(chips * self.code, self.n_fft)[bins]) ** 2
            k = int(np.argmax(power))
            score = float(power[k] / energy)
            if best is None or score > best[0]:
                best = score, offset, float(self.freqs[bins[k]])
        # Eight coherent chip equivalents suppress random correlation peaks.
        threshold = max(
            8 / n,
            (
                0.06
                if self.config.mode == "periodic"
                else 0.4 * self.config.power_fraction
            ),
        )
        if best is not None and best[0] >= threshold:
            self.frequency = 0.5 * self.frequency + 0.5 * best[2]
            self.timing = best[1]
            self.updates += 1


def demodulate_packet(
    rx_audio,
    preamble_code,
    data_codebook,
    rrc_taps,
    max_doppler_hz=100.0,
    *,
    return_details=False,
    pilot_tracking=None,
):
    """Decode one v3 packet, discovering pilot settings from its checked header.

    Default result is (payload_or_None, sync_idx, initial_vfo_hz). Details expose
    tracking diagnostics. Tries preamble candidates until a complete CRC passes.
    """
    tracking = PILOT_TRACKING if pilot_tracking is None else pilot_tracking

    def finish(result):
        return (
            result
            if return_details
            else (result.payload, result.sync_idx, result.vfo_hz)
        )

    rx_audio = np.asarray(rx_audio, dtype=float)
    if rx_audio.ndim != 1 or not np.all(np.isfinite(rx_audio)):
        raise ValueError("Receiver input must be a finite mono waveform.")
    if not np.isfinite(max_doppler_hz) or not 0 <= max_doppler_hz < CHIP_RATE / 2:
        raise ValueError("Frequency search must be in [0, CHIP_RATE/2).")
    if M_ARY != 16 or np.shape(data_codebook) != (16, len(preamble_code)):
        raise ValueError("Frame v3 requires 16 data codes matching the preamble.")
    samples_per_sym = len(preamble_code) * SPS
    preamble_span = PREAMBLE_LEN * samples_per_sym
    max_search_idx = (
        len(rx_audio) - preamble_span - MIN_FRAME_BYTES * 2 * samples_per_sym
    )
    if max_search_idx < 0:
        return finish(DecodeResult(None, "too_short"))
    if not np.any(rx_audio):
        return finish(DecodeResult(None, "no_preamble"))
    t = np.arange(len(rx_audio)) / float(FS)
    bb_complex = rx_audio * np.exp(-1j * 2 * np.pi * F_SUB * t) * 2
    bb_filtered = signal.lfilter(rrc_taps, 1.0, bb_complex)
    n_fft = max(2048, 1 << int(np.ceil(np.log2(len(preamble_code) * 4))))
    freqs = np.fft.fftfreq(n_fft, d=1.0 / CHIP_RATE)
    valid_bins = np.where(np.abs(freqs) <= max_doppler_hz)[0]
    fine_bins = np.where(np.abs(freqs) <= 10.0)[0]
    chip_times = np.arange(len(preamble_code)) / CHIP_RATE

    def preamble_metric(idx):
        chips = bb_filtered[idx : idx + preamble_span].reshape(
            PREAMBLE_LEN, samples_per_sym
        )[:, ::SPS]
        energy = np.sum(np.abs(chips) ** 2, axis=1) * len(preamble_code)
        spec = (
            np.abs(np.fft.fft(chips * preamble_code, n=n_fft, axis=1)[:, valid_bins])
            ** 2
        )
        scores = np.divide(
            spec, energy[:, None], out=np.zeros_like(spec), where=energy[:, None] > 0
        )
        k = int(np.argmax(scores.mean(axis=0)))
        return float(scores[:, k].mean()), float(freqs[valid_bins[k]]), scores[:, k]

    def decode_symbol(start, frequency, templates):
        if start < 0 or start + samples_per_sym > len(bb_filtered):
            return None
        chips = bb_filtered[start : start + samples_per_sym : SPS]
        chips = chips * np.exp(-1j * 2 * np.pi * frequency * chip_times)
        spec = np.abs(np.fft.fft(templates * chips, n=n_fft, axis=1)[:, fine_bins])
        return int(np.argmax(np.max(spec, axis=1)))

    step = max(1, SPS // 2)
    starts = np.arange(0, max_search_idx + 1, step)
    scores = np.array([preamble_metric(int(idx))[0] for idx in starts])
    # Keep separate peaks, including endpoints; a continuous pilot can be
    # stronger than the real preamble after fading, so do not take only argmax.
    peaks, _ = signal.find_peaks(
        np.r_[-np.inf, scores, -np.inf],
        height=MIN_PREAMBLE_SCORE,
        distance=max(1, SPS // step),
    )
    candidates = sorted(peaks - 1, key=lambda k: scores[k], reverse=True)
    fallback = DecodeResult(None, "no_preamble")
    checked_starts = set()
    for k in candidates:
        coarse = int(starts[k])
        refined = [
            (preamble_metric(idx), idx)
            for idx in range(
                max(0, coarse - step), min(max_search_idx, coarse + step) + 1
            )
        ]
        (score, frequency, repeats), start = max(refined, key=lambda item: item[0][0])
        if start in checked_starts:
            continue
        checked_starts.add(start)
        if (
            score < MIN_PREAMBLE_SCORE
            or np.count_nonzero(repeats >= MIN_REPEAT_SCORE) < PREAMBLE_LEN - 1
        ):
            continue
        if fallback.status == "no_preamble":
            fallback = DecodeResult(None, "invalid_header", start, frequency, score)
        header_symbols = []
        for offset in range(HEADER_BYTES * 2):
            symbol = decode_symbol(
                start + (PREAMBLE_LEN + offset) * samples_per_sym,
                frequency,
                data_codebook,
            )
            header_symbols.append(symbol)
            # Reject false pilot peaks cheaply, before decoding the full header.
            if offset < 6 and offset % 2:
                byte = (header_symbols[-2] << 4) | header_symbols[-1]
                if byte != FRAME_MARKER[offset // 2]:
                    break
        if len(header_symbols) != HEADER_BYTES * 2:
            continue
        header = bytes(
            (header_symbols[i] << 4) | header_symbols[i + 1]
            for i in range(0, len(header_symbols), 2)
        )
        parsed = parse_header(header)
        if parsed is None:
            continue
        payload_len, config = parsed
        schedule = body_schedule((payload_len + CRC_BYTES) * 2, config)
        body_start = start + (PREAMBLE_LEN + HEADER_BYTES * 2) * samples_per_sym
        tracker = PilotTracker(
            frequency, max_doppler_hz, payload_pilot_code(preamble_code), config, n_fft
        )
        observations = 0

        def result(payload, status, raw=None):
            return DecodeResult(
                payload,
                status,
                start,
                frequency,
                score,
                raw,
                config,
                observations,
                tracker.updates,
                tracker.frequency,
                tracker.timing,
            )

        if body_start + len(schedule) * samples_per_sym > len(bb_filtered):
            if fallback.pilot_config is None:
                fallback = result(None, "truncated")
            continue
        templates, _ = body_templates(preamble_code, data_codebook, config)
        body_symbols = []
        for position, data_index in enumerate(schedule):
            nominal = body_start + position * samples_per_sym
            if data_index is None or config.mode == "continuous":
                observations += 1
                if tracking:
                    tracker.observe(bb_filtered, nominal)
            if data_index is None:
                continue
            symbol = decode_symbol(
                nominal + tracker.timing, tracker.frequency, templates
            )
            if symbol is None:
                break
            body_symbols.append(symbol)
        if len(body_symbols) != (payload_len + CRC_BYTES) * 2:
            if fallback.pilot_config is None:
                fallback = result(None, "truncated")
            continue
        body = bytes(
            (body_symbols[i] << 4) | body_symbols[i + 1]
            for i in range(0, len(body_symbols), 2)
        )
        frame = header + body
        payload, status = validate_frame(frame)
        decoded = result(payload, status, body[:-CRC_BYTES])
        if decoded.ok:
            return finish(decoded)
        if fallback.pilot_config is None:
            fallback = decoded
    return finish(fallback)


# ==============================================================================
# 6. WAV I/O HELPERS & MONTE CARLO BER SWEEP
# ==============================================================================
def save_wav(filename, audio_float, *, float_wav=False):
    """Save PCM16 for playback, or unclipped float32 for simulated channels."""
    audio_float = np.asarray(audio_float)
    if float_wav:
        samples = audio_float.astype(np.float32)
    else:
        samples = np.int16(np.clip(audio_float, -1.0, 1.0) * 32767)
    wavfile.write(filename, FS, samples)


def load_wav(filename):
    fs_in, pcm = wavfile.read(filename)
    if fs_in != FS:
        raise ValueError(f"Expected {FS} Hz WAV, got {fs_in} Hz.")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if np.issubdtype(pcm.dtype, np.floating):
        return pcm.astype(float)
    if pcm.dtype == np.uint8:
        return (pcm.astype(float) - 128.0) / 128.0
    if np.issubdtype(pcm.dtype, np.signedinteger):
        return pcm.astype(float) / float(2 ** (np.iinfo(pcm.dtype).bits - 1))
    raise ValueError(f"Unsupported WAV sample dtype: {pcm.dtype}")


def timing_summary(label, elapsed_s, audio_s):
    """RTF < 1 means faster than real time; audio includes file silence/tails."""
    rtf = elapsed_s / audio_s if audio_s > 0 else float("nan")
    speed = audio_s / elapsed_s if elapsed_s > 0 else float("inf")
    return (
        f"[TIMING {label}] Compute: {elapsed_s:.6f} s | Audio: {audio_s:.3f} s | "
        f"RTF: {rtf:.4f} | Speed: {speed:.2f}x real time"
    )


def payload_rates(attempted_bits, delivered_bits, tx_seconds):
    """Payload bps over total transmitted WAV time, including failed attempts.

    Includes preamble/header/CRC, pilots, filter tail and file silence; excludes
    compute time, simulated RX-only delay, ACKs and external turnaround time.
    """
    if tx_seconds <= 0:
        return float("nan"), float("nan")
    return attempted_bits / tx_seconds, delivered_bits / tx_seconds


def run_ber_sweep(preamble_code, data_codebook, rrc_taps):
    rng = np.random.default_rng(12345)
    payload_len_bytes = 16
    sym_duration_ms = (len(preamble_code) / CHIP_RATE) * 1000.0
    bit_rate = (BITS_PER_SYM * CHIP_RATE) / len(preamble_code)
    proc_gain_db = 10.0 * np.log10(len(preamble_code))

    print(pilot_summary(configured_pilot(), len(preamble_code), payload_len_bytes))
    print(f"Pilot tracking: {PILOT_TRACKING}")
    print("=" * 72)
    print(
        f"16-ARY GOLD CODE SSB MODEM V{SOFTWARE_VERSION} | N={len(preamble_code)} chips"
    )
    print(
        f"Symbol Duration: {sym_duration_ms:.1f} ms | Uncoded Symbol Bit Rate: {bit_rate:.2f} bps | Proc Gain: +{proc_gain_db:.1f} dB"
    )
    print(
        f"VFO Offset: {VFO_OFFSET_HZ:+.1f} Hz | Trials/SNR: {TRIALS_PER_SNR} ({payload_len_bytes*8} bits/pkt)"
    )
    print("=" * 72)
    print(
        f"{'SNR (2.4kHz)':>12} | {'Raw BER*':>10} | {'Packet OK':>10} | {'BER bits':>8} | {'Avg VFO Err':>12} | {'Payload bps':>11} | {'Goodput bps':>11}"
    )
    print("-" * 72)

    for snr_db in SNR_SWEEP_DB:
        total_bits = 0
        total_bit_errors = 0
        packets_ok = 0
        attempted_payload_bits = 0
        delivered_payload_bits = 0
        vfo_errors = []
        encode_times, decode_times = [], []
        tx_durations, rx_durations = [], []

        for _ in range(TRIALS_PER_SNR):
            tx_bytes = rng.integers(
                0, 256, size=payload_len_bytes, dtype=np.uint8
            ).tobytes()
            attempted_payload_bits += 8 * len(tx_bytes)
            started = perf_counter()
            audio_tx, _, _ = modulate_packet(
                tx_bytes, preamble_code, data_codebook, rrc_taps
            )
            encode_times.append(perf_counter() - started)
            tx_durations.append(len(audio_tx) / FS)
            jitter = int(rng.integers(0, 120))
            rx_audio = pass_through_ssb_channel(
                audio_tx, snr_db, VFO_OFFSET_HZ, jitter, rng
            )
            started = perf_counter()
            decoded = demodulate_packet(
                rx_audio,
                preamble_code,
                data_codebook,
                rrc_taps,
                MAX_DOPPLER_HZ,
                return_details=True,
            )
            decode_times.append(perf_counter() - started)
            rx_durations.append(len(rx_audio) / FS)
            rx_bytes, est_vfo = decoded.payload, decoded.vfo_hz

            # Conditional raw BER: only complete candidates of the expected
            # length, including CRC failures. Acquisition/header failures are
            # packet failures, not fabricated all-zero received bits.
            if decoded.raw_payload is not None and len(decoded.raw_payload) == len(
                tx_bytes
            ):
                tx_bits = np.unpackbits(np.frombuffer(tx_bytes, dtype=np.uint8))
                rx_bits = np.unpackbits(
                    np.frombuffer(decoded.raw_payload, dtype=np.uint8)
                )
                total_bits += len(tx_bits)
                total_bit_errors += int(np.sum(tx_bits != rx_bits))
            if decoded.ok and rx_bytes == tx_bytes:
                packets_ok += 1
                delivered_payload_bits += 8 * len(tx_bytes)
            if decoded.status not in ("no_preamble", "too_short"):
                vfo_errors.append(abs(est_vfo - VFO_OFFSET_HZ))

        ber = total_bit_errors / total_bits if total_bits else float("nan")
        pkt_rate = f"{packets_ok}/{TRIALS_PER_SNR}"
        avg_vfo_err = float(np.mean(vfo_errors)) if vfo_errors else float("nan")
        payload_bps, goodput_bps = payload_rates(
            attempted_payload_bits, delivered_payload_bits, sum(tx_durations)
        )
        print(
            f"{snr_db:>9.1f} dB | {ber:>10.5f} | {pkt_rate:>10} | {total_bits:>8} | {avg_vfo_err:>9.2f} Hz | {payload_bps:>11.3f} | {goodput_bps:>11.3f}"
        )
        for label, times, durations in (
            ("ENC", encode_times, tx_durations),
            ("DEC", decode_times, rx_durations),
        ):
            if times:
                print(
                    "  "
                    + timing_summary(label + " aggregate", sum(times), sum(durations))
                )
                print(
                    f"    Mean/packet: {np.mean(times):.6f} s | "
                    f"Max/packet: {max(times):.6f} s | "
                    f"Worst packet RTF: {max(t / d for t, d in zip(times, durations)):.4f}"
                )
    print(
        "Payload/goodput bps use total TX WAV time, including padding and failed attempts."
    )
    print(
        "Goodput counts only CRC-accepted, exactly matching payloads; ACK/turnaround not modeled."
    )
    print("* Raw BER is conditional on complete candidates with the expected length.")
    print("=" * 72)


if __name__ == "__main__":
    preamble_code, data_codebook = generate_gold_codes(LFSR_DEGREE, M_ARY)
    rrc_taps = rrc_filter(SPS, RRC_SPAN, ROLLOFF)

    print(f"Gold SSB modem V{SOFTWARE_VERSION} | Wire format {FRAME_MARKER[-1]}")
    print(
        "Timing uses wall-clock compute time, excluding WAV I/O and channel simulation."
    )
    print("RTF = compute/audio duration (<1 is faster than real time).")
    print("Audio duration includes padding; batch throughput is not streaming latency.")
    # 1. Single WAV Loopback Demonstration
    msg = b"CQ CQ DE K6TEST - 16-ARY GOLD CODE SSB LOOPBACK"
    print(pilot_summary(configured_pilot(), len(preamble_code), len(msg)))
    started = perf_counter()
    tx = modulate_packet(
        msg, preamble_code, data_codebook, rrc_taps, return_details=True
    )
    tx_wav_audio, papr_raw, papr_cfr = tx.audio, tx.papr_before_db, tx.papr_after_db
    encode_elapsed = perf_counter() - started
    print(timing_summary("ENC", encode_elapsed, len(tx_wav_audio) / FS))
    print(
        f"[POWER] Composite chips before normalization: {tx.composite_chip_power:.6f} | "
        f"Active audio mean square after CFR/scaling: {tx.active_audio_power:.6f}"
    )
    save_wav("tx_mary_ssb.wav", tx_wav_audio)
    print(
        f"[WAV TX] Saved 'tx_mary_ssb.wav' | PAPR: {papr_raw:.2f} dB -> {papr_cfr:.2f} dB"
    )

    rx_wav_audio = pass_through_ssb_channel(
        load_wav("tx_mary_ssb.wav"), snr_db=-12.0, vfo_offset_hz=VFO_OFFSET_HZ
    )
    save_wav("rx_mary_ssb.wav", rx_wav_audio, float_wav=True)
    rx_input = load_wav("rx_mary_ssb.wav")
    started = perf_counter()
    decoded = demodulate_packet(
        rx_input,
        preamble_code,
        data_codebook,
        rrc_taps,
        MAX_DOPPLER_HZ,
        return_details=True,
    )
    decode_elapsed = perf_counter() - started
    print(timing_summary("DEC", decode_elapsed, len(rx_input) / FS))
    decoded_msg, sync_idx, est_vfo = decoded.payload, decoded.sync_idx, decoded.vfo_hz
    decoded_text = (
        decoded_msg.decode("utf-8", errors="replace")
        if decoded_msg is not None
        else "[packet rejected]"
    )
    print(
        f"[WAV RX] Status: {decoded.status} | Sync @ sample {sync_idx} | Est VFO: {est_vfo:+.2f} Hz"
    )
    print(
        f"[PILOT RX] Mode: {decoded.pilot_config.mode if decoded.pilot_config else 'unknown'} | "
        f"Observations/updates: {decoded.pilot_observations}/{decoded.tracking_updates} | "
        f"Final VFO: {decoded.final_vfo_hz:+.2f} Hz | Timing shift: {decoded.timing_offset_samples} samples"
    )
    print(f"[WAV RX] Decoded: {decoded_text}\n")
    delivered_bits = 8 * len(msg) if decoded.ok and decoded_msg == msg else 0
    payload_bps, goodput_bps = payload_rates(
        8 * len(msg), delivered_bits, len(tx_wav_audio) / FS
    )
    print(
        f"[RATE] Payload: {payload_bps:.3f} bps | Goodput: {goodput_bps:.3f} bps "
        "(TX WAV duration including padding; no ACK/turnaround)"
    )

    # 2. Optional Monte Carlo BER vs. SNR Sweep
    if RUN_SNR_SWEEP:
        run_ber_sweep(preamble_code, data_codebook, rrc_taps)
