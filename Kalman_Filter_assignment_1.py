import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initial Conditions
a = 1  # Acceleration [m/s^2]
# if acceleration changes it can be seperately calculated by using the measured position and time and/or velocity and time.
x_0 = 0  # Position [m]
v_0 = 0  # Velocity [m/s]
t = 0.1  # Time step [s]

# Initial process erros in process covariance matrix
# The process covariance matrix corrects itself in the process of the recursive algorithm, however we should initially make an educated guess of what the error in the first predicted state will be
# In this example, measurements consist only of position data
x_err_process = 0  # Position error [m]
v_err_process = 0  # Velocity error [m/s]

# Errors in measurement covariance matrix
x_err_measure = 10  # Position error [m]

# Control variable matrix
U = np.array([a])

# Step 1 Initialize process covariance matrix
# The covariance elements in the process covariance matrix are set to 0, based on the assumption that variable 'x' is independent of the other variable 'v'
# No adjustments are made to the estimates of one variable due to the process error of the other variable
def init_process_covariance_matrix(x_err_process, v_err_process):
    P = np.array([[x_err_process ** 2, 0], [0, v_err_process ** 2]])
    return P


# Step 2 Calculate the (new) predicted state
# State noise - predicted errors due to external factors influencing the system; (external forces acting on the object, e.g. friction, wind gust, water currents), but are known not to exceed certain values
def calculate_state_estimate(prev_state, control_variable_matrix, noise_matrix):
    """Returns a new state estimate based on the previous state, control variables, and process noise"""

    A = np.array([[1, t], [0, 1]])
    B = np.array([[0.5 * (t ** 2)], [t]])

    state_estimate = A.dot(prev_state) + B.dot(control_variable_matrix) + noise_matrix
    return state_estimate


# Step 3 Calculate the (new) predicted process covariance matrix (error in the state estimate)
# Process noise - keeps the covariance matrix from becoming too small or 0
def calculate_process_covariance_estimate(prev_process_covariance, noise_matrix):
    A = np.array([[1, t], [0, 1]])
    process_covariance_estimate = A @ prev_process_covariance @ A.transpose() + noise_matrix
    # Zero out the covariance terms in the matrix
    return np.diag(np.diag(process_covariance_estimate))


# Step 4 Calculate the measurement covariance matrix (error in the state measurement)
def calculate_measurement_covariance(x_err_measure):
    R = np.array([x_err_measure ** 2])
    return R


# Step 5 Calculate the Kalman gain (weighting factor)
# In this example, the kalman gain matrix should have 2x1 format, because the measurement covariance is 1x1
def calculate_kalman_gain(process_covariance_estimate, measurement_covariance):
    # General use case when process and measurement covariance matrices have the same format:
    # H = np.identity(2)
    # denominator = H @ process_covariance_estimate @ H.transpose() + measurement_covariance
    # kalman_gain = process_covariance_estimate @ H @ np.linalg.inv(denominator)

    # Adapted for the assignment:
    kalman_gain = process_covariance_estimate.take(0).item() / (
        process_covariance_estimate.take(0).item() + measurement_covariance.item()
    )
    # print(process_covariance_estimate.take(0).item())
    return np.array([kalman_gain, 0]).transpose()


# Step 6 Calculate the (new) measurement
# Measurement noise - errors due to unknown factors of the equipment, temperature variations, filter inaccuracies, etc. and can not be predicted, but are known not to exceed certain values
def calculate_state_measurement(measured_state, noise_matrix):
    """Returns a new state measurement based on the measured state measurement and measurement noise"""

    C = np.array([1, 0]).transpose()

    state_measurement = C.dot(measured_state) + noise_matrix
    # print(state_measurement.shape)
    return state_measurement


# Step 7 Calculate adjusted state, based on the predicted state, measured state, and kalman gain
def calculate_state(state_estimate, state_measurement, kalman_gain):
    """Returns an adjustate state based on the kalman gain and the difference between the estimate and measurement"""

    H = np.identity(2)
    adjusted_state = state_estimate + kalman_gain * np.array([state_measurement.take(0) - (H @ state_estimate).take(0)])
    return adjusted_state


# Step 8 Update the process covariance matrix
def calculate_process_covariance(kalman_gain, process_covariance_estimate):
    """Returns an adjusted process covariance matrix based on the kalman gain"""

    I = np.identity(2)
    H = np.identity(2)
    process_covariance = (I - kalman_gain.transpose() @ H) @ process_covariance_estimate
    # print(process_covariance, process_covariance_estimate)
    return np.diag(np.diag(process_covariance))


# Step 9 Update prev state & covariance matrix


def main():
    data_frame = pd.read_excel("KF_data.xlsx", engine="openpyxl")
    print(np.take(data_frame["data"].to_numpy(), -1))

    prev_process_covariance = init_process_covariance_matrix(x_err_process, v_err_process)
    prev_state = np.array([x_0, v_0]).transpose()

    collected_data = data_frame["data"].to_numpy()

    time = np.linspace(0, 100.1, num=1001)
    kalman_corrected_data = []

    i = 0
    for x_measurement in collected_data:
        if i > len(collected_data):
            break
        # Predict the state
        state_estimate = calculate_state_estimate(prev_state, control_variable_matrix=U, noise_matrix=0)

        # Predict the error in the estimate
        process_covariance_estimate = calculate_process_covariance_estimate(prev_process_covariance, noise_matrix=0)

        # Make a state measurement
        state_measurement = calculate_state_measurement(
            measured_state=np.array([x_measurement, 0]).transpose(), noise_matrix=0
        )

        # Calculate the error in the measurement
        measurement_covariance = calculate_measurement_covariance(x_err_measure)

        # Calculate the weigthing factor (kalman gain)
        gain = calculate_kalman_gain(process_covariance_estimate, measurement_covariance)

        # Calculate the (new) adjusted state, based on the estimate and the measurement
        state_new = calculate_state(state_estimate, state_measurement, gain)

        # Calculate the (new) adjusted error in the estimate, based on the weigthing factor
        process_covariance_new = calculate_process_covariance(gain, process_covariance_estimate)

        # Update the previous state and process covariance
        prev_state = state_new
        prev_process_covariance = process_covariance_new
        i += 1
        kalman_corrected_data.append(state_new.take(0).item())

        print("Observed: ", x_measurement, "      ", "KF adjusted: ", state_new)

    plt.plot(time, collected_data, label="Measurement data")
    plt.plot(time, kalman_corrected_data, label="KF corrected data")
    plt.xlabel("Time [s]")
    plt.xticks(np.arange(0, 100.1, step=10))
    plt.ylabel("Position [m]")
    plt.title("Kalman Filter 1D example")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
