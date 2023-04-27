import time
import numpy as np
import board
import busio
from mpu6050 import mpu6050

'''
Generated by ChatGPT
'''

class EKF:
    def __init__(self, x0, P0, Q, R):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(self, u, dt):
        roll, pitch, x_dot, y_dot = self.x
        accel_x, accel_y, accel_z, gyro_x, gyro_y = u

        # Calculate the state transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Calculate the control input matrix
        B = np.array([
            [0.5 * dt**2, 0, 0, 0],
            [0, 0.5 * dt**2, 0, 0],
            [dt, 0, 0, 0],
            [0, dt, 0, 0]
        ])

        # Calculate the process noise covariance matrix
        G = np.array([
            [0.5 * dt**2, 0, 0, 0],
            [0, 0.5 * dt**2, 0, 0],
            [dt, 0, 0, 0],
            [0, dt, 0, 0]
        ])
        Qd = G @ self.Q @ G.T

        # Calculate the predicted state and covariance
        x_pred = F @ self.x + B @ np.array([accel_x, accel_y, 0, 0]) + np.array([0, 0, x_dot, y_dot])
        P_pred = F @ self.P @ F.T + Qd

        self.x = x_pred
        self.P = P_pred

    def update(self, y):
        roll, pitch, x_dot, y_dot = self.x
        accel_x, accel_y, accel_z, x_dot_meas, y_dot_meas = y

        # Calculate the measurement matrix
        H = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Calculate the measurement noise covariance matrix
        R = self.R.copy()
        R[:3, :3] += np.diag([x_dot**2, y_dot**2, 0])

        # Calculate the Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Calculate the innovation
        y_pred = H
