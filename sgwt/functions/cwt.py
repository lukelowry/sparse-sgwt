import numpy as np

def gaussian_wavelet(time, a=1, b=0, w0=1):
    '''
    Will also normalize by dt, assuming first two values in time can be used to determine this
    '''

    # Shifted and Scaled Time 
    t = (time - b)/a
    dt = time[1] - time[0]

    # Normalization 
    norm_const = (dt/a) * np.pi**(-0.25)

    # Gaussian distribution
    gauss = np.exp(-(t**2/2))

    # Non-local oscillation
    ac = np.exp(1j*w0*t)-np.exp(-w0**2/2)

    # Wavelet
    wavelet = (gauss*ac)*norm_const

    return wavelet 