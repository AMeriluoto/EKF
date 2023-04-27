import EKF
from mpu6050 import mpu6050

# Initialize the MPU-6050 sensor
sensor = mpu6050(0x68)

# Set up the EKF
dt = 0.1  # Time step
Q = np.diag([0.01, 0.01, 0.001, 0.001])  # Process noise covariance
R = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # Measurement noise covariance
ekf = EKF(np.array([0, 0, 0, 0]), np.diag([1, 1, 1, 1]), Q, R)

# Loop to estimate velocity
while True:
    # Read the accelerometer and gyroscope data from the sensor
    accel_data = sensor.get_accel_data()
    gyro_data = sensor.get_gyro_data()

    # Convert the data to numpy arrays
    accel = np.array([accel_data['x'], accel_data['y'], accel_data['z']])
    gyro = np.array([gyro_data['x'], gyro_data['y'], gyro_data['z']])

    # Run the EKF prediction step
    ekf.predict(np.concatenate((accel, gyro)), dt)

    # Run the EKF update step
    ekf.update(np.concatenate((accel, gyro)))

    # Get the estimated velocity from the EKF state vector
    vel = ekf.x[:2]

    # Print out the estimated velocity
    print("Estimated velocity: ({:.2f}, {:.2f})".format(vel[0], vel[1]))
