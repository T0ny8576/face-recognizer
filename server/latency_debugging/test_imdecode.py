import os
import time

import numpy as np
import cv2

from plot_cpu_freq import plot_timeline

SERVER_DIR = os.path.dirname(os.path.realpath(__file__))


def test_imdecode(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    _, jpg_data = cv2.imencode('.jpg', img)
    np_data = np.frombuffer(jpg_data, dtype=np.uint8)
    trial_count = 200
    wall_clock_time = np.zeros((trial_count,), dtype=np.float32)
    cpu_time = np.zeros((trial_count,), dtype=np.float32)

    for i in range(trial_count):
        t002 = time.time()
        t002_cpu = time.process_time()
        # cpu_freq_start = psutil.cpu_freq().current
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        # cpu_freq_end = psutil.cpu_freq().current
        t003 = time.time()
        t003_cpu = time.process_time()
        wall_clock_time[i] = t003 - t002
        cpu_time[i] = t003_cpu - t002_cpu

    print("Wall clock time:")
    print(list(wall_clock_time))
    print("CPU time:")
    print(list(cpu_time))

    sample_py_wall_time = 1000. * np.array(wall_clock_time)
    sample_py_cpu_time = 1000. * np.array(cpu_time)
    plot_timeline(np.arange(len(sample_py_wall_time)), sample_py_wall_time, "py_jpg_decode_time_wall.png")
    plot_timeline(np.arange(len(sample_py_cpu_time)), sample_py_cpu_time, "py_jpg_decode_time_cpu.png")


if __name__ == "__main__":
    jpg_file = os.path.join(SERVER_DIR, "sample.jpg")
    test_imdecode(jpg_file)
