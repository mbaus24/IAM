import numpy as np
from numpy.linalg import norm
from scipy.linalg import circulant
from scipy.ndimage import gaussian_filter, convolve1d


def get_gaussian_filter(std, filter_len):
    x = np.zeros(filter_len)
    x[0] = 1
    h = gaussian_filter(x, sigma=std, mode='wrap')
    return convolve1d(h, x, mode='wrap')


def filter2matrix(h):
    h_shift = np.concatenate((h[len(h)//2:], h[:len(h)//2]))
    return circulant(h_shift)


def convolve(x, h):
    return convolve1d(x, h, mode='wrap')

def add_noise(x, snr):
    noise = np.random.randn(x.size)
    return x + 10 ** (-snr / 20) * norm(x) / norm(noise) * noise
    

def compute_snr(x_true, x_noisy):
    return 20 * np.log10(norm(x_true) / norm(x_true - x_noisy))


def soft_thresholding(x, tau):
    pass  # à compléter


def ista_l1(f, grad_f, x0, step_size, alpha, n_iterations, return_iterates):
    pass  # à compléter
