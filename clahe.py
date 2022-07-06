import operator
from functools import reduce

import numpy as np
import tqdm

dtype = np.uint16
max_dtype_value = np.iinfo(dtype).max
available_clip_sizes = ((64, 64),)

frames = np.fromfile("3.data", dtype=dtype).reshape((250, 512, 640))

limit = input("Input limit: ")

if limit.isdigit():
    limit = int(limit)
else:
    limit = None

txt_available_clip_sizes = "\n".join(
    [f"{idx}. {el}" for idx, el in enumerate(available_clip_sizes, start=1)]
)
clip_size = available_clip_sizes[
    int(input(f"Clip sizes: \n {txt_available_clip_sizes}\n")) - 1
]

processed_frames = []


def divide_into_clips(frame, clip_size):
    i_range = int(frame.shape[0] / clip_size[0])
    j_range = int(frame.shape[1] / clip_size[1])
    clips = []
    for i in range(i_range):
        clips.append([])
        for j in range(j_range):
            clips[-1].append(
                frame[
                    i * clip_size[0] : (i + 1) * clip_size[0],
                    j * clip_size[1] : (j + 1) * clip_size[1],
                ]
            )
    return clips


def get_hist(frame):
    zeros_hist = np.zeros((max_dtype_value + 1,))

    for line in frame:
        for byte in line:
            zeros_hist[byte] += 1
    return zeros_hist


def get_r_func(hist: np.ndarray, total_bytes):
    return hist / total_bytes


def get_limited_hist(hist, limit):
    if limit is not None:
        return np.clip(hist, a_min=None, a_max=limit)
    return hist


def get_cumulative_line(r_func):
    return np.cumsum(r_func)


def get_trunslate_array(cumulative_line):
    translate_array = cumulative_line * max_dtype_value
    return translate_array


def translate_clip(frame, translate_dict):
    copied_frame = np.copy(frame)
    for i in range(copied_frame.shape[0]):
        for j in range(copied_frame.shape[1]):
            copied_frame[i, j] = translate_dict[copied_frame[i, j]]
    return copied_frame


def equilize_clip(clip, limit):
    total_bytes = reduce(operator.mul, clip.shape)
    hist = get_hist(clip)
    limited_hist = get_limited_hist(hist, limit)
    r_func = get_r_func(limited_hist, total_bytes)
    cumulative_line = get_cumulative_line(r_func)
    translate_dict = get_trunslate_array(cumulative_line)
    translated_clip = translate_clip(clip, translate_dict)
    return translated_clip


def process_frame(frame, limit, clip_size):
    copied_frame = np.copy(frame)
    total_bytes = reduce(operator.mul, copied_frame.shape)
    clips = divide_into_clips(copied_frame, clip_size)

    equilized_clips = []

    for row in clips:
        equilized_clips.append([])
        for clip in row:
            equilized_clip = equilize_clip(clip, limit)
            equilized_clips[-1].append(equilized_clip)

    return

for frame in tqdm.tqdm(frames):
    processed_frame = process_frame(frame, limit, clip_size)
    processed_frames.append(processed_frame)

with open("frames.data", "wb") as f:
    np.save(f, processed_frames)
