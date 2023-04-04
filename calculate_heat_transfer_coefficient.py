import pandas as pd
from matplotlib import pyplot as plt

# Read the CSV file
filename = 'heat_coeff_estimation_2.csv'
data = pd.read_csv(filename)

# Constants
efficiency = 0.20  # 20% gross efficiency
weight_kg = 67  # weight_kg in kg
height_cm = 176  # Height in cm

# Function to calculate skin surface area using the DuBois formula
def skin_surface_area(height_cm, weight_kg):
    return 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)

surface_area = skin_surface_area(height_cm, weight_kg)

# Function to calculate heat transfer coefficient
def calculate_coefficients(row):
    P_running = row['power']
    T_skin = row['skin_temperature']
    T_core = row['core_temperature']

    # Calculate metabolic heat production
    M = P_running / efficiency

    # Calculate heat transfer
    Q = M - P_running

    # Calculate heat transfer coefficient
    h = Q / (surface_area * (T_core - T_skin))

    return h

# Function to estimate power given the heat transfer coefficient
def estimate_power(row):
    T_skin = row['skin_temperature']
    T_core = row['core_temperature']
    h = data['heat_transfer_coefficient'].mean()

    # Calculate heat transfer
    Q = h * surface_area * (T_core - T_skin)

    # Calculate total heat production
    M = Q /(1 - efficiency)

    # Calculate power
    P = M * efficiency

    return P

# Apply the calculate_coefficients function to each row in the data
data['heat_transfer_coefficient'] = data.apply(calculate_coefficients, axis=1, result_type='expand')

# Apply the estimate_power function to each row in the data
data['estimated_power'] = data.apply(estimate_power, axis=1, result_type='expand')

# Save the data with calculated values to a new CSV file
output_filename = 'output_with_coefficients.csv'
data.to_csv(output_filename, index=False)

print("Heat transfer coefficients calculated and saved to", output_filename)

# Calculate and print the average heat transfer coefficient
average_coefficient = data['heat_transfer_coefficient'].mean()
print("Average heat transfer coefficient:", average_coefficient)

# Calculate and print the standard deviation of the heat transfer coefficient
std_coefficient = data['heat_transfer_coefficient'].std()
print("Standard deviation of heat transfer coefficient:", std_coefficient)

# Plot estimated power vs power on a scatter plot
plt.scatter(data['power'], data['estimated_power'])
plt.xlabel('Power (W)')
plt.ylabel('Estimated power (W)')
plt.title('Estimated power vs power')
plt.show()




