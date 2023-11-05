import numpy as np
from scipy.fft import fft, ifft
from scipy import signal as sig

_float_type = np.float64
_int_type = np.int64


def decimate(s, d):
    l = len(s)
    s_ = s[np.mod(np.arange(l) * d, l)]
    return s_


def is_balanced(s, tolerance=1):
    return abs(np.sum(s) - np.sum(~s)) <= tolerance


def preferred_pair(N, d=3):
    s = sig.max_len_seq(N)[0]
    s_ = decimate(s, d)
    return s, s_


def gold_sequence_set(ps0, ps1, n, balanced_only=True):
    g = []
    for i in range(len(ps1)):
        s = np.logical_xor(ps0, np.roll(ps1, i))
        if balanced_only:
            if is_balanced(s):
                g.append(s)
        else:
            g.append(s)
        if len(g) >= n:
            break
    else:
        raise ValueError(
            f"{len(g)}/{i} balanced sequences found. Try a different preferred pair."
        )

    return np.row_stack(g)


def gold_codec(N, size):
    s0, s1 = preferred_pair(N, 5)
    gs_matrix = gold_sequence_set(s0, s1, size)
    return np.float32(gs_matrix * 2 - 1)


def cyclic_xcorr(s1, s2):
    return ifft(fft(s1) * fft(s2).conj)


def cyclic_xcorr2(s1, s2):
    s22 = np.tile(s2, 2)[:-1]
    xc = np.round(np.correlate(s1, s22, mode="valid")).astype(_int_type)
    return xc


def cxc_values(S1, S2):
    xc = np.round(ifft(S1 * S2.conj()).real).astype(np.int64)
    vc = np.unique(xc, return_counts=True)
    df = pd.DataFrame(zip(*vc), columns=("value", "count"))
    return df
