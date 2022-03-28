import numpy as np

def pad_array(array, pad_value, top=0, right=0, bottom=0, left=0):
    padded = array.copy()
    h, w = padded.shape

    if right > 0:
        pad_r = [[pad_value] * h for _ in range(right)]
        padded = np.insert(padded, w, pad_r, axis=1)

    if left > 0:
        pad_l = [[pad_value] * h for _ in range(left)]
        padded = np.insert(padded, 0, pad_l, axis=1)

    h, w = padded.shape
    if bottom > 0:
        pad_b = [[pad_value] * w for _ in range(bottom)]
        padded = np.insert(padded, h, pad_b, axis=0)

    if top > 0:
        pad_t = [[pad_value] * w for _ in range(top)]
        padded = np.insert(padded, 0, pad_t, axis=0)

    return padded
