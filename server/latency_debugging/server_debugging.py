import argparse
import logging
import os
import pickle
import time

import numpy as np
import cv2
import torch
from torch.autograd import Variable

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

import openface

import psutil

SOURCE_NAME = 'openface'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 2
DNN_ONES_SIZE = (1, 3, 96, 96)
THRESHOLD = 0.6
BB_COLOR = (0, 255, 0)
LABEL_TEXT_COLOR = (0, 0, 0)
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.6
LABEL_FONT_THICKNESS = 1

SERVER_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SERVER_DIR, 'models')
DEFAULT_DLIB_FACE_PREDICTOR_PATH = os.path.join(DEFAULT_MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
DEFAULT_DLIB_FACE_DETECTOR_PATH = os.path.join(DEFAULT_MODEL_DIR, 'mmod_human_face_detector.dat')
DEFAULT_OPENFACE_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'nn4.small2.v1.pt')
DEFAULT_CLASSIFIER_PATH = os.path.join(DEFAULT_MODEL_DIR, 'classifier.pkl')

PROFILING = False
PROFILING_COUNT = 500

global logger


class FaceEngine(cognitive_engine.Engine):
    def __init__(self, dlib_face_predictor, dlib_face_detector, openface_model, classifier,
                 upsampling, multi_face=False, cpu=False):
        self._device = 'cpu' if cpu else 'cuda'
        self._multi_face = multi_face
        self.dlib_model = openface.AlignDlib(dlib_face_predictor, dlib_face_detector, upsample=upsampling)
        self.openface_model = openface.OpenFaceNet()
        if self._device == 'cuda':
            self.openface_model.load_state_dict(torch.load(openface_model, map_location='cuda'))
            self.openface_model.to(torch.device('cuda'))
        else:
            self.openface_model.load_state_dict(torch.load(openface_model))
        self.openface_model.eval()
        with open(classifier, 'rb') as classifier_file:
            self.le, self.clf = pickle.load(classifier_file)
        torch.set_grad_enabled(False)

        self._g = None
        self._static_input = None
        self._static_output = None
        if self._device == 'cuda':
            # Warm up the GPU
            ones = torch.ones(DNN_ONES_SIZE, dtype=torch.float32).to(torch.device(self._device))
            self._static_input = Variable(ones)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._static_output = self.openface_model(self._static_input)
            torch.cuda.current_stream().wait_stream(s)
            # Capture CUDA graph
            self._g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._g):
                self._static_output = self.openface_model(self._static_input)

        # Profiling
        self.t0 = 0.
        self.t1 = 0.
        self.t2 = 0.
        self.t3 = 0.
        self.t4 = 0.
        self.t5 = 0.
        self.last_time = time.time()
        self.count = 0
        self.logfile = None
        self.logtext = ""
        self.fixed_img_data = None ######################################################
        self.align_time = [] ######################################################
        self.t003_list = [] ######################################################
        self.t003_cpu_list = []  ######################################################
        self.cpu_freq_list = []  ######################################################

    def model_forward(self, x):
        if self._device == 'cuda':
            # Replay CUDA graph
            x = Variable(x)
            self._static_input.copy_(x)
            self._g.replay()
            res = self._static_output.data[0]
            return res
        else:
            return self.openface_model(x)

    def get_reps(self, rgb_img, img_dim=96, multiple=False):
        detect_t1 = time.time()
        self.t1 = detect_t1
        # logger.debug('  + Original size: {}'.format(rgb_img.shape))

        if multiple:
            bbs = self.dlib_model.getAllFaceBoundingBoxes(rgb_img)
        else:
            bb1 = self.dlib_model.getLargestFaceBoundingBox(rgb_img)
            bbs = [bb1]
        if len(bbs) == 0 or bbs[0] is None:
            # logger.warning('Unable to find a face.')
            return []
        detect_t2 = time.time()
        self.t2 = detect_t2
        # logger.debug('Face detection took {:.3f} seconds.'.format(detect_t2 - detect_t1))

        reps = []
        for bb in bbs:
            align_t3 = time.time()
            aligned_face = self.dlib_model.align(img_dim, rgb_img, bb,
                                                 landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is None:
                # logger.warning('Unable to predict face landmarks.')
                continue

            align_t4 = time.time()
            # logger.debug('This bbox is centered at {}, {}'.format(bb.center().x, bb.center().y))
            # logger.debug('Alignment took {:.3f} seconds.'.format(align_t4 - align_t3))

            self.align_time.append((align_t4 - align_t3) * 1000)  #############################################################
            aligned_face = (aligned_face / 255.).astype(np.float32)
            aligned_face = np.expand_dims(np.transpose(aligned_face, (2, 0, 1)), axis=0)  # BCHW order
            aligned_face = torch.from_numpy(aligned_face)
            aligned_face = aligned_face.to(torch.device(self._device))
            rep = self.model_forward(aligned_face)
            rep = rep.cpu().detach().numpy()

            reps_t5 = time.time()
            # logger.debug('Neural network forward pass took {:.3f} seconds.'.format(reps_t5 - align_t4))
            reps.append(((bb.left(), bb.top(), bb.right(), bb.bottom()), rep))
        sorted_reps = sorted(reps, key=lambda x: x[0][0])
        self.t3 = time.time()
        return sorted_reps

    def handle(self, input_frame):
        self.t0 = time.time()

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)
        if len(input_frame.payloads) == 0:
            # A new client has arrived
            self.count = 0
            self.logtext = ""
            self.logfile = os.path.join(SERVER_DIR, "Server-Log-" + str(int(self.t0)) + ".txt")
            return result_wrapper

        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        if self.fixed_img_data is None: #############################################################
            img_data = input_frame.payloads[0]
            self.fixed_img_data = img_data
            # img = cv2.imread(os.path.join(SERVER_DIR, "sample.jpg"), cv2.IMREAD_COLOR)
            # _, img_data = cv2.imencode('.jpg', img)
            # self.fixed_img_data = img_data
            # #############################################################
            # np_data = np.frombuffer(img_data, dtype=np.uint8)
            # img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            # cv2.imwrite(os.path.join(SERVER_DIR, "sample.jpg"), img)
            # #############################################################

        else:
            img_data = self.fixed_img_data #############################################################

        np_data = np.frombuffer(img_data, dtype=np.uint8)
        # for xxi in range(200):  #############################################################
        t002 = time.time()  #############################################################
        t002_cpu = time.process_time()  #############################################################
        # cpu_freq_start = psutil.cpu_freq().current
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        # cpu_freq_end = psutil.cpu_freq().current
        t003 = time.time() #############################################################
        t003_cpu = time.process_time()  #############################################################
        self.t003_list.append((t003 - t002) * 1000) #############################################################
        self.t003_cpu_list.append((t003_cpu - t002_cpu) * 1000)  #############################################################
        # self.cpu_freq_list.append((cpu_freq_start + cpu_freq_end) / 2.)  #############################################################
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        reps = self.get_reps(img, multiple=self._multi_face)
        face_results = []
        if len(reps) > 1:
            pass # logger.info('List of faces in image from left to right:')
        for face in reps:
            pred_t6 = time.time()
            bb = face[0]
            rep = face[1].reshape(1, -1)

            predictions = self.clf.predict_proba(rep).ravel()
            max_ind = np.argmax(predictions)
            person = str(self.le.inverse_transform([max_ind])[0])
            confidence = predictions[max_ind]
            pred_t7 = time.time()
            # logger.debug('Prediction took {:.3f} seconds.'.format(pred_t7 - pred_t6))
            if self._multi_face:
                pass # logger.info('Predict {} @ x={} with {:.2f} confidence.'.format(person, bb[0], confidence))
            else:
                pass # logger.info('Predict {} with {:.2f} confidence.'.format(person, confidence))

            if confidence < THRESHOLD:
                person = 'unknown'
            img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), BB_COLOR, 2)
            label = '{}: {:.0%}'.format(person, confidence)
            face_results.append(label)
            (label_w, label_h), baseline = cv2.getTextSize(label, LABEL_FONT,
                                                           LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)
            label_left, label_top = bb[0], bb[1] - label_h - baseline
            label_right, label_bottom = bb[0] + label_w, bb[1]
            img = cv2.rectangle(img, (label_left, label_top), (label_right, label_bottom), BB_COLOR, -1)
            img = cv2.putText(img, label, (label_left, label_bottom - baseline), LABEL_FONT,
                              LABEL_FONT_SCALE, LABEL_TEXT_COLOR, LABEL_FONT_THICKNESS)

        self.t4 = time.time()
        img_to_send = img
        img_to_send = cv2.cvtColor(img_to_send, cv2.COLOR_RGB2BGR)
        _, jpeg_img = cv2.imencode('.jpg', img_to_send)
        img_data = jpeg_img.tobytes()

        # Return the annotated image, or the original image if no faces were found
        if len(face_results) > 0:
            face_results_text = ','.join(face_results)
            result = gabriel_pb2.ResultWrapper.Result()
            result.payload_type = gabriel_pb2.PayloadType.TEXT
            result.payload = face_results_text.encode()
            result_wrapper.results.append(result)

        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data
        result_wrapper.results.append(result)
        self.t5 = time.time()

        global PROFILING
        if PROFILING:
            self.count += 1
            pre = (self.t1 - self.t0) * 1000
            detect = (self.t2 - self.t1) * 1000
            repr = (self.t3 - self.t2) * 1000
            infer = (self.t4 - self.t3) * 1000
            post = (self.t5 - self.t4) * 1000
            wait = (self.t0 - self.last_time) * 1000
            # fps = 1.0 / (self.t5 - self.last_time)
            self.logtext += ("#{}, time = {}, done = {}, pre = {:.3f} ms, detect = {:.3f} ms, repr = {:.3f} ms, "
                             "infer = {:.3f} ms, post = {:.3f} ms, wait = {:.3f} ms\n"
                             .format(self.count, self.t0, self.t5, pre, detect, repr, infer, post, wait))

            if self.count == PROFILING_COUNT:
                with open(self.logfile, "a") as logfile:
                    logfile.write(self.logtext)
                print("Log written to file.")
                self.logtext = ""
                print("t003")  #############################################################
                print(np.var(self.t003_list))  #############################################################
                print([(self.t003_list[ti], self.t003_cpu_list[ti]) for ti in range(len(self.t003_list))])  #############################################################
                print("Align time")  #############################################################
                print(self.align_time)  #############################################################
                print(np.var(self.align_time))  #############################################################
            self.last_time = self.t5

        return result_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlib_face_predictor_path', type=str, default=DEFAULT_DLIB_FACE_PREDICTOR_PATH)
    parser.add_argument('--dlib_face_detector_type', type=str, choices=['HOG', 'CNN'], default='CNN')
    parser.add_argument('--dlib_face_detector_path', type=str, default=DEFAULT_DLIB_FACE_DETECTOR_PATH)
    parser.add_argument('--upsample', type=int, default=1)
    parser.add_argument('--openface_model_path', type=str, default=DEFAULT_OPENFACE_MODEL_PATH)
    parser.add_argument('--classifier_path', type=str, default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument('--multi_face', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--profiling', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)

    if args.dlib_face_detector_type == 'HOG':
        args.dlib_face_detector_path = None
    if args.profiling:
        global PROFILING
        PROFILING = True
        cv2.setNumThreads(0)  # Disable multithreading for profiling

    def engine_factory():
        return FaceEngine(args.dlib_face_predictor_path, args.dlib_face_detector_path, args.openface_model_path,
                          args.classifier_path, args.upsample, args.multi_face, args.cpu)

    local_engine.run(engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()
