import operator
import os.path
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from numba import jit

dtype = np.uint16
max_dtype_value = np.iinfo(dtype).max

frames = np.fromfile("3.data", dtype=dtype).reshape((250, 512, 640))

limit = input("Input limit: ")
if limit.isdigit():
    limit = int(limit)
else:
    limit = None

processed_frames = []


@jit()
def get_hist(frame):
    zeros_hist = np.zeros((max_dtype_value + 1,))

    for line in frame:
        for byte in line:
            zeros_hist[byte] += 1
    return zeros_hist


def save_hist(hist, file_name):
    plt.hist(hist)
    plt.savefig(os.path.join("hists", file_name))


@jit()
def get_r_func(hist: np.ndarray, total_bytes):
    return hist / total_bytes


@jit()
def get_limited_hist(hist, limit):
    if limit is not None:
        return np.clip(hist, a_min=None, a_max=limit)
    return hist


@jit()
def get_cumulative_line(r_func):
    return np.cumsum(r_func)


@jit()
def get_trunslate_dict(cumulative_line):
    return {
        idx: round(cumulative_line[idx] * max_dtype_value)
        for idx in range(len(cumulative_line))
    }


@jit()
def translate_frame(frame, translate_dict):
    copied_frame = np.copy(frame)
    for i in range(copied_frame.shape[0]):
        for j in range(copied_frame.shape[1]):
            copied_frame[i, j] = translate_dict[copied_frame[i, j]]
    return copied_frame


def process_frame(frame, idx, limit=None):
    total_bytes = reduce(operator.mul, frame.shape)
    hist = get_hist(frame)
    save_hist(hist, f"hist_{idx}.png")
    limited_hist = get_limited_hist(hist, limit)
    save_hist(limited_hist, f"limited_hist_{idx}.png")
    r_func = get_r_func(limited_hist, total_bytes)
    cumulative_line = get_cumulative_line(r_func)
    translate_dict = get_trunslate_dict(cumulative_line)
    translated_frame = translate_frame(frame, translate_dict)
    return translated_frame


for idx, frame in tqdm.tqdm(list(enumerate(frames))):
    processed_frame = process_frame(frame, idx, limit)
    processed_frames.append(processed_frame)

with open("frames.data", "wb") as f:
    np.save(f, processed_frames)
