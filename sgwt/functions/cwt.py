import numpy as np


def gaussian_wavelet(time, a=1, b=0, w0=1):
    """
    Compute a Morlet-like Gaussian wavelet.

    Generates a complex-valued wavelet with Gaussian envelope and oscillatory
    carrier, normalized for continuous wavelet transform applications.

    Parameters
    ----------
    time : ndarray
        Time vector of shape ``(n_time,)`` in seconds. Must have at least
        2 elements for timestep inference.
    a : float, default=1
        Scale parameter (temporal dilation). Larger values produce wider
        wavelets centered on lower frequencies.
    b : float, default=0
        Translation parameter (center time in seconds).
    w0 : float, default=1
        Central angular frequency in rad/s. Default of ``2*np.pi`` gives
        1 Hz center frequency.

    Returns
    -------
    ndarray
        Complex-valued wavelet of shape ``(n_time,)``, normalized by ``dt/a``.

    Examples
    --------
    >>> import numpy as np
    >>> from sgwt.functions import gaussian_wavelet
    >>> t = np.linspace(0, 10, 1000)
    >>> psi = gaussian_wavelet(t, a=0.5, b=5.0, w0=2*np.pi)
    """
    # Shifted and Scaled Time
    t = (time - b) / a
    dt = time[1] - time[0]

    # Normalization
    norm_const = (dt / a) * np.pi**(-0.25)

    # Gaussian distribution
    gauss = np.exp(-(t**2 / 2))

    # Non-local oscillation
    ac = np.exp(1j * w0 * t) - np.exp(-w0**2 / 2)

    # Wavelet
    wavelet = (gauss * ac) * norm_const

    return wavelet