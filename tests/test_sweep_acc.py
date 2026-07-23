import numpy as np

from mkidgen3.drivers.sweepacc import sums_to_mean_rms


def _golden(samples):
    """samples: (n, N, 2) int16 -> (mean, rms) the way gen3d does it today."""
    c = samples[..., 0].astype(np.float64) + 1j * samples[..., 1].astype(np.float64)
    mean = c.mean(axis=0)
    rms = c.real.std(axis=0) + 1j * c.imag.std(axis=0)
    return mean, rms


def _sums(samples):
    """Build the (N, 4) int64 sums array the gateware would produce."""
    i = samples[..., 0].astype(np.int64)
    q = samples[..., 1].astype(np.int64)
    return np.stack([i.sum(0), q.sum(0), (i * i).sum(0), (q * q).sum(0)], axis=1)


def test_sums_to_mean_rms_matches_numpy():
    rng = np.random.default_rng(42)
    samples = rng.integers(-32768, 32768, size=(4096, 2048, 2), dtype=np.int16)
    mean, rms = sums_to_mean_rms(_sums(samples), 4096)
    gmean, grms = _golden(samples)
    np.testing.assert_allclose(mean, gmean, rtol=1e-12)
    np.testing.assert_allclose(rms, grms, rtol=1e-9)


def test_sums_to_mean_rms_constant_signal_zero_rms():
    samples = np.full((100, 8, 2), 1234, dtype=np.int16)
    mean, rms = sums_to_mean_rms(_sums(samples), 100)
    np.testing.assert_allclose(mean, np.full(8, 1234 + 1234j))
    # var computed as E[x^2]-E[x]^2 can go epsilon negative: must clamp, not NaN
    assert np.all(np.isfinite(rms))
    np.testing.assert_allclose(rms, np.zeros(8), atol=1e-6)


def test_sums_to_mean_rms_extreme_values_no_overflow():
    # full scale for the max software average must not overflow the float math
    n = 2**20
    sums = np.array([[np.int64(-32768) * n, np.int64(32767) * n,
                      np.int64(32768) ** 2 * n, np.int64(32767) ** 2 * n]])
    mean, rms = sums_to_mean_rms(sums, n)
    np.testing.assert_allclose(mean, [-32768 + 32767j])
    np.testing.assert_allclose(rms, [0 + 0j], atol=1e-3)
