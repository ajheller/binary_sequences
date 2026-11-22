"""Gold sequence construction and periodic correlation helpers.

Sequences occupy rows; phases are cyclic right shifts. Binary samples are 0/1,
and bipolar samples are -1/+1. Correlations are unnormalized full-period sums.
"""

from itertools import chain, count, islice
from math import gcd
from operator import index

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, ifft

_float_type = np.float64
_int_type = np.int64


def _degree(n):
    n = index(n)
    if n < 3 or n % 4 == 0:
        raise ValueError("n must be at least 3 and not divisible by 4")
    return n


def t(n):
    """Return the peak absolute correlation bound for a Gold family."""
    n = _degree(n)
    return 2 ** ((n + (1 if n % 2 else 2)) // 2) + 1


def theta(n):
    """Return the three signed cross-correlation levels."""
    bound = t(n)
    return -bound, -1, bound - 2


def Rxy(n):
    """Return preferred-pair correlation levels and their occurrence counts."""
    n = _degree(n)
    if n % 2:
        base, offset = 2 ** (n - 2), 2 ** ((n - 3) // 2)
        middle = 2 ** (n - 1) - 1
    else:
        base, offset = 2 ** (n - 3), 2 ** ((n - 4) // 2)
        middle = 3 * 2 ** (n - 2) - 1
    bound = t(n)
    return {-1: middle, -bound: base - offset, bound - 2: base + offset}


def q_generator(n, q_fn=1):
    """Yield sufficient preferred-pair decimations in the tutorial's order.

    q_fn selects 2**k + 1 (1), 2**(2*k) - 2**k + 1 (2), or a callable.
    Custom callables must eventually repeat an odd residue to terminate.
    """
    n = _degree(n)
    if q_fn == 1:
        q_fn = lambda k: 2**k + 1
    elif q_fn == 2:
        q_fn = lambda k: 2 ** (2 * k) - 2**k + 1
    if not callable(q_fn):
        raise TypeError("q_fn must be 1, 2, or a callable")
    period = 2**n - 1
    good_gcd = 1 if n % 2 else 2
    seen = set()
    for k in count():
        if gcd(n, k) != good_gcd:
            continue
        q = index(q_fn(k)) % period
        if q % 2:
            if q in seen:
                return
            seen.add(q)
            yield q


def generate_good_qs(n):
    """Chain both decimation families; equivalent factors may repeat."""
    return chain(q_generator(n, 1), q_generator(n, 2))


def nth(iterable, n, default=None):
    """Return the zero-based nth item of an iterable."""
    return next(islice(iterable, n, None), default)


def decimate_sequence(s, q):
    """Return len(s) samples of cyclic decimation by integer q."""
    s = np.asarray(s)
    if s.ndim != 1 or not s.size:
        raise ValueError("s must be a nonempty one-dimensional sequence")
    q = index(q) % len(s)
    return s[(np.arange(len(s)) * q) % len(s)]


def decimate(s, d):
    """Compatibility name for decimate_sequence."""
    return decimate_sequence(s, d)


def characteristic_phase(s):
    """Return the right shift at which decimation by two leaves s unchanged."""
    s = np.asarray(s)
    if s.ndim != 1 or not s.size:
        raise ValueError("s must be a nonempty one-dimensional sequence")
    for p in range(len(s)):
        shifted = np.roll(s, p)
        if np.array_equal(shifted, decimate_sequence(shifted, 2)):
            return p
    raise ValueError("No characteristic phase found")


def is_sequence_bipolar(s):
    """Return whether a nonempty array contains only -1 and +1."""
    s = np.asarray(s)
    return bool(s.size and np.all(np.isin(s, [-1, 1])))


def to_bipolar(s):
    """Map binary samples to signed -1/+1 samples, including unsigned input."""
    s = np.asarray(s)
    if not np.all(np.isin(s, [0, 1])):
        raise ValueError("s must contain binary 0/1 samples")
    return s.astype(np.int8) * 2 - 1


def is_sequence_balanced(s, tolerance=1):
    """Compare symbol counts for a binary or bipolar sequence."""
    s = np.asarray(s)
    if s.ndim != 1 or not s.size:
        raise ValueError("s must be a nonempty one-dimensional sequence")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if not (np.all(np.isin(s, [0, 1])) or is_sequence_bipolar(s)):
        raise ValueError("s must be binary or bipolar")
    ones = int(np.count_nonzero(s == 1))
    return abs(2 * ones - s.size) <= tolerance


def is_balanced(s, tolerance=1):
    """Compatibility name for is_sequence_balanced."""
    return is_sequence_balanced(s, tolerance)


def preferred_pair(N, d=3):
    """Generate an order-N m-sequence and its d-decimation.

    N is register length, not sequence period. The legacy default is d=3;
    callers supplying d are responsible for choosing a preferred decimation.
    """
    N = _degree(N)
    s = sig.max_len_seq(N)[0]
    return s, decimate_sequence(s, d)


def _family_rows(s1, s2, size, include_pair, initial_phase, balanced_only):
    size, initial_phase = index(size), index(initial_phase)
    minimum = 2 if include_pair else 1
    maximum = len(s2) + (2 if include_pair else 0)
    if not minimum <= size <= maximum:
        raise ValueError(f"set_size must be between {minimum} and {maximum}")
    rows = [s1, s2] if include_pair else []
    if len(rows) == size:
        return np.vstack(rows)
    for p in range(initial_phase, initial_phase + len(s2)):
        candidate = np.logical_xor(s1, np.roll(s2, p))
        if balanced_only and not is_sequence_balanced(candidate):
            continue
        rows.append(candidate)
        if len(rows) == size:
            return np.vstack(rows)
    raise ValueError(f"Only {len(rows)} eligible sequences available; requested {size}")


def gold_code_set(n, set_size, q=None, bipolar=True, include_pair=True,
                  initial_phase=0, balanced_only=False):
    """Generate a subset of a Gold family, with sequences in rows.

    q=None selects the tutorial's first sufficient preferred decimation.
    Explicit q must form a preferred pair. initial_phase is an integer right
    shift, defaulting to zero. balanced_only filters XOR candidates; the
    original pair, when included, occupies the first two rows.
    """
    n = _degree(n)
    if q is None:
        q = next(generate_good_qs(n))
    s1, s2 = preferred_pair(n, q)
    rows = _family_rows(s1, s2, set_size, include_pair, initial_phase,
                        balanced_only).astype(np.int8)
    return to_bipolar(rows) if bipolar else rows


def gold_sequence_set(ps0, ps1, n, balanced_only=True):
    """Legacy API: return n boolean XOR members from a supplied binary pair."""
    ps0, ps1 = np.asarray(ps0), np.asarray(ps1)
    if ps0.ndim != 1 or ps1.shape != ps0.shape or not ps0.size:
        raise ValueError("The pair must contain equal-length nonempty sequences")
    if not (np.all(np.isin(ps0, [0, 1])) and np.all(np.isin(ps1, [0, 1]))):
        raise ValueError("The pair must contain binary 0/1 samples")
    return _family_rows(ps0, ps1, n, False, 0, balanced_only).astype(bool)


def gold_codec(N, size):
    """Legacy codec: q=5, balanced XOR members only, bipolar float32 output."""
    return gold_code_set(N, size, q=5, include_pair=False,
                         balanced_only=True).astype(np.float32)


def cyclic_xcorr(s1, s2):
    """Periodic correlation at right-shift lags for real equal-length inputs."""
    s1, s2 = np.asarray(s1), np.asarray(s2)
    if s1.ndim != 1 or s1.shape != s2.shape or not s1.size:
        raise ValueError("Inputs must be equal-length nonempty sequences")
    return ifft(fft(s1) * fft(s2).conj()).real


def cyclic_xcorr2(s1, s2):
    """Direct periodic correlation, with the same lag convention as cyclic_xcorr."""
    s1, s2 = np.asarray(s1), np.asarray(s2)
    if s1.ndim != 1 or s1.shape != s2.shape or not s1.size:
        raise ValueError("Inputs must be equal-length nonempty sequences")
    # Widen integer inputs before summation to prevent int8 overflow.
    s1, s2 = s1.astype(np.float64), s2.astype(np.float64)
    return np.array([np.dot(s1, np.roll(s2, p))
                     for p in range(len(s1))]).round().astype(_int_type)


def _rows(x_set, axis):
    rows = np.asarray(x_set)
    if rows.ndim != 2 or not all(rows.shape):
        raise ValueError("x_set must be a nonempty two-dimensional array")
    rows = np.moveaxis(rows, axis, -1)
    if not is_sequence_bipolar(rows):
        raise ValueError("All sequences must be bipolar (-1 and 1)")
    return rows


def cyclic_xcorr_all(x_set, axis=1):
    """Return [sequence, sequence, lag] correlations; axis selects samples."""
    X = fft(_rows(x_set, axis), axis=1)
    result = np.empty((len(X), len(X), X.shape[1]))
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            result[i, j] = ifft(x * y.conj()).real
    return result


def cyclic_xcorr_all_map(x_set):
    """Map-based variant for row-oriented bipolar sequences."""
    X = fft(_rows(x_set, 1), axis=1)
    pairs = ((x, y) for x in X for y in X)
    result = list(map(lambda pair: ifft(pair[0] * pair[1].conj()).real, pairs))
    return np.asarray(result).reshape(len(X), len(X), -1)


def cyclic_xcorr_all_parallel(gc_set, n_jobs=None, verbose=0):
    """Joblib variant for row-oriented bipolar sequences."""
    from joblib import Parallel, delayed, cpu_count

    if n_jobs is None:
        n_jobs = cpu_count()
    X = fft(_rows(gc_set, 1), axis=1)
    result = Parallel(n_jobs=n_jobs, verbose=verbose, prefer="threads")(
        delayed(lambda a, b: ifft(a * b.conj()).real)(x, y)
        for x in X for y in X
    )
    return np.asarray(result).reshape(len(X), len(X), -1)


def cxc_values(S1, S2):
    """Return a pandas table of correlation levels/counts from Fourier spectra."""
    import pandas as pd

    xc = np.rint(ifft(np.asarray(S1) * np.asarray(S2).conj()).real).astype(np.int64)
    values, counts = np.unique(xc, return_counts=True)
    return pd.DataFrame({"value": values, "count": counts})
