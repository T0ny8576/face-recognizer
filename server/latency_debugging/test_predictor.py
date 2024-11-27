import os
import time
import gc

import numpy as np
import cv2
import dlib
import psutil

from plot_cpu_freq import plot_timeline

SERVER_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SERVER_DIR, 'models')
DEFAULT_DLIB_FACE_PREDICTOR_PATH = os.path.join(DEFAULT_MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
DEFAULT_DLIB_FACE_DETECTOR_PATH = os.path.join(DEFAULT_MODEL_DIR, 'mmod_human_face_detector.dat')

predictor = dlib.shape_predictor(DEFAULT_DLIB_FACE_PREDICTOR_PATH)
detector = dlib.cnn_face_detection_model_v1(DEFAULT_DLIB_FACE_DETECTOR_PATH)
CNN_DETECTOR_CONF_THRESHOLD = 0.5


def test_predictor(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bbs = [mmod_rect.rect for mmod_rect in detector(rgb_img, 0) if mmod_rect.confidence > CNN_DETECTOR_CONF_THRESHOLD]
    assert len(bbs) == 1
    bb = bbs[0]

    trial_count = 20000
    wall_clock_time = np.zeros((trial_count,), dtype=np.float32)
    cpu_time = np.zeros((trial_count,), dtype=np.float32)

    for i in range(trial_count):
        # gc.disable()
        t00 = time.time()
        t00_cpu = time.process_time()
        points = predictor(rgb_img, bb)
        t01 = time.time()
        t01_cpu = time.process_time()
        wall_clock_time[i] = (t01 - t00) * 1000.
        cpu_time[i] = (t01_cpu - t00_cpu) * 1000.
        if cpu_time[i] > 10.:
            print("Frame: {}".format(i))
            print(cpu_time[i])
            # print(psutil.cpu_freq().current)
        # gc.enable()
        # gc.collect()

    sample_py_wall_time = np.array(wall_clock_time)
    sample_py_cpu_time = np.array(cpu_time)
    plot_timeline(np.arange(len(sample_py_wall_time)), sample_py_wall_time, "py_predictor_time_wall.png")
    plot_timeline(np.arange(len(sample_py_cpu_time)), sample_py_cpu_time, "py_predictor_time_cpu.png")


if __name__ == "__main__":
    print(psutil.cpu_freq().current)
    jpg_file = os.path.join(SERVER_DIR, "sample.jpg")
    test_predictor(jpg_file)
    print(psutil.cpu_freq().current)
