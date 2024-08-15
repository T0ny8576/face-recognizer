import argparse
import logging
import os
import pickle
import time

import numpy as np
import cv2
import torch

from gabriel_server import cognitive_engine
from gabriel_server import local_engine
from gabriel_protocol import gabriel_pb2

import openface

SOURCE_NAME = 'openface'
INPUT_QUEUE_MAXSIZE = 60
PORT = 9099
NUM_TOKENS = 2
DNN_ONES_SIZE = (1, 3, 96, 96)
THRESHOLD = 0.4

SERVER_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SERVER_DIR, 'models')
DEFAULT_DLIB_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
DEFAULT_OPENFACE_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'nn4.small2.v1.pt')
DEFAULT_CLASSIFIER_PATH = os.path.join(DEFAULT_MODEL_DIR, 'classifier.pkl')

global logger


class FaceEngine(cognitive_engine.Engine):
    def __init__(self, dlib_model_path, openface_model_path, classifier_path, multi_face=False, cpu=False):
        self._device = 'cpu' if cpu else 'cuda'
        self._multi_face = multi_face
        self.dlib_model = openface.AlignDlib(dlib_model_path)
        self.openface_model = openface.OpenFaceNet()
        if self._device == 'cuda':
            self.openface_model.load_state_dict(torch.load(openface_model_path, map_location='cuda'))
            self.openface_model.to(torch.device('cuda'))
        else:
            self.openface_model.load_state_dict(torch.load(openface_model_path))
        self.openface_model.eval()
        with open(classifier_path, 'rb') as classifier_file:
            self.le, self.clf = pickle.load(classifier_file)

        # Warm up the GPU
        ones = torch.ones(DNN_ONES_SIZE, dtype=torch.float32).to(torch.device(self._device))
        self.openface_model(ones)

        self.t0 = 0
        self.t1 = 0
        self.t2 = 0
        self.t3 = 0

    def get_reps(self, rgb_img, img_dim=96, multiple=False):
        self.t0 = time.time()
        logger.debug('  + Original size: {}'.format(rgb_img.shape))

        if multiple:
            bbs = self.dlib_model.getAllFaceBoundingBoxes(rgb_img)
        else:
            bb1 = self.dlib_model.getLargestFaceBoundingBox(rgb_img)
            bbs = [bb1]
        if len(bbs) == 0 or bbs[0] is None:
            logger.warning('Unable to find a face.')
            return []
        self.t1 = time.time()
        logger.debug('Face detection took {:.3f} seconds.'.format(self.t1 - self.t0))

        reps = []
        for bb in bbs:
            aligned_face = self.dlib_model.align(img_dim, rgb_img, bb,
                                                 landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is None:
                logger.warning('Unable to find a face.')
                return []

            self.t2 = time.time()
            logger.debug('This bbox is centered at {}, {}'.format(bb.center().x, bb.center().y))
            logger.debug('Alignment took {:.3f} seconds.'.format(self.t2 - self.t1))
            self.t1 = self.t2

            aligned_face = (aligned_face / 255.).astype(np.float32)
            aligned_face = np.expand_dims(np.transpose(aligned_face, (2, 0, 1)), axis=0)  # BCHW order
            aligned_face = torch.from_numpy(aligned_face)
            aligned_face = aligned_face.to(torch.device(self._device))
            rep = self.openface_model(aligned_face)
            rep = rep.cpu().detach().numpy()

            self.t2 = time.time()
            logger.debug('Neural network forward pass took {:.3f} seconds.'.format(self.t2 - self.t1))
            self.t1 = self.t2
            reps.append((bb.center().x, rep))
        sorted_reps = sorted(reps, key=lambda x: x[0])
        return sorted_reps

    def handle(self, input_frame):
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        result_wrapper = cognitive_engine.create_result_wrapper(status)

        if len(input_frame.payloads) == 0:
            return result_wrapper

        img_data = input_frame.payloads[0]
        np_data = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        reps = self.get_reps(img)
        if len(reps) > 1:
            logger.info('List of faces in image from left to right:')
        for face in reps:
            bbx = face[0]
            rep = face[1].reshape(1, -1)

            predictions = self.clf.predict_proba(rep).ravel()
            max_ind = np.argmax(predictions)
            person = str(self.le.inverse_transform([max_ind])[0])
            confidence = predictions[max_ind]
            self.t3 = time.time()
            logger.debug('Prediction took {:.3f} seconds.'.format(self.t3 - self.t2))
            self.t2 = self.t3
            if self._multi_face:
                logger.info('Predict {} @ x={} with {:.2f} confidence.'.format(person, bbx, confidence))
            else:
                logger.info('Predict {} with {:.2f} confidence.'.format(person, confidence))

            # TODO: Draw bounding box with labels and conf on the image using OpenCV if conf > thres
            # cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(im, 'Moth Detected', (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
            img_to_send = img
            img_to_send = cv2.cvtColor(img_to_send, cv2.COLOR_RGB2BGR)
            _, jpeg_img = cv2.imencode('.jpg', img_to_send)
            img_data = jpeg_img.tobytes()

        # Return the annotated image, or the original image if no faces were found
        result = gabriel_pb2.ResultWrapper.Result()
        result.payload_type = gabriel_pb2.PayloadType.IMAGE
        result.payload = img_data
        result_wrapper.results.append(result)
        return result_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlib_model_path', type=str, default=DEFAULT_DLIB_MODEL_PATH)
    parser.add_argument('--openface_model_path', type=str, default=DEFAULT_OPENFACE_MODEL_PATH)
    parser.add_argument('--classifier_path', type=str, default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument('--multi_face', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)

    def engine_factory():
        return FaceEngine(args.dlib_model_path, args.openface_model_path, args.classifier_path,
                          args.multi_face, args.cpu)

    local_engine.run(engine_factory, SOURCE_NAME, INPUT_QUEUE_MAXSIZE, PORT, NUM_TOKENS)


if __name__ == '__main__':
    main()