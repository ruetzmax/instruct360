# based on https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html

import numpy as np
import cv2 as cv

from src.util import normalize_rotation_matrix, normalize_translation_vector


def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    R = normalize_rotation_matrix(rotation_matrix).astype(np.float32)

    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)

    pitch = np.arcsin(sy)
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw], dtype=np.float32)


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)


def init_kalman(dt: float = 0.125) -> cv.KalmanFilter:
    n_states = 18
    n_measurements = 6
    n_inputs = 0

    kf = cv.KalmanFilter(n_states, n_measurements, n_inputs)

    # setup transition matrix
    kf.transitionMatrix = np.eye(n_states, dtype=np.float32)

    kf.transitionMatrix[0, 3] = dt
    kf.transitionMatrix[1, 4] = dt
    kf.transitionMatrix[2, 5] = dt
    kf.transitionMatrix[3, 6] = dt
    kf.transitionMatrix[4, 7] = dt
    kf.transitionMatrix[5, 8] = dt
    kf.transitionMatrix[0, 6] = 0.5 * (dt ** 2)
    kf.transitionMatrix[1, 7] = 0.5 * (dt ** 2)
    kf.transitionMatrix[2, 8] = 0.5 * (dt ** 2)

    kf.transitionMatrix[9, 12] = dt
    kf.transitionMatrix[10, 13] = dt
    kf.transitionMatrix[11, 14] = dt
    kf.transitionMatrix[12, 15] = dt
    kf.transitionMatrix[13, 16] = dt
    kf.transitionMatrix[14, 17] = dt
    kf.transitionMatrix[9, 15] = 0.5 * (dt ** 2)
    kf.transitionMatrix[10, 16] = 0.5 * (dt ** 2)
    kf.transitionMatrix[11, 17] = 0.5 * (dt ** 2)
    
    #setup measurement matrix
    kf.measurementMatrix = np.zeros((n_measurements, n_states), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1.0
    kf.measurementMatrix[1, 1] = 1.0
    kf.measurementMatrix[2, 2] = 1.0
    kf.measurementMatrix[3, 9] = 1.0 
    kf.measurementMatrix[4, 10] = 1.0
    kf.measurementMatrix[5, 11] = 1.0

    kf.processNoiseCov = np.eye(n_states, dtype=np.float32) * 1e-5
    kf.measurementNoiseCov = np.eye(n_measurements, dtype=np.float32) * 1e-4
    kf.errorCovPost = np.eye(n_states, dtype=np.float32)

    # kf.statePost = np.zeros((n_states, 1), dtype=np.float32)

    return kf


def do_kalman_step(
    kf: cv.KalmanFilter,
    rotation_matrix,
    translation_vector,
):

    R_meas = normalize_rotation_matrix(rotation_matrix).astype(np.float32)
    t_meas = normalize_translation_vector(translation_vector).astype(np.float32)

    eulers_meas = _rotation_matrix_to_euler(R_meas)

    measurement = np.zeros((6, 1), dtype=np.float32)
    measurement[0, 0] = t_meas[0]
    measurement[1, 0] = t_meas[1]
    measurement[2, 0] = t_meas[2]
    measurement[3, 0] = eulers_meas[0]
    measurement[4, 0] = eulers_meas[1]
    measurement[5, 0] = eulers_meas[2]

    kf.predict()
    state = kf.correct(measurement)

    t_filt = state[0:3, 0]
    roll_f, pitch_f, yaw_f = state[9, 0], state[10, 0], state[11, 0]
    R_filt = _euler_to_rotation_matrix(roll_f, pitch_f, yaw_f)

    return R_filt, t_filt
