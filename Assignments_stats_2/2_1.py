import matplotlib.pyplot as plt
import numpy as np
# Given data
companies = ["Allied Signal", "Bankers Trust", "General Mills", "ITT Industries", "J.P.Morgan & Co.",
             "Lehman Brothers", "Marriott", "MCI", "Merrill Lynch", "Microsoft", "Morgan Stanley",
             "Sun Microsystems", "Travelers", "US Airways", "Warner-Lambert"]
measure_x = [24.23, 25.53, 25.41, 24.14, 29.62, 28.25, 25.81, 24.39, 40.26, 32.95, 91.36, 25.99, 39.42, 26.71, 35.00]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.bar(companies, measure_x, color='blue')
plt.xlabel('Company')
plt.ylabel('Measure X (%)')
plt.title('Measure X for Each Company')
plt.xticks(rotation=90)
plt.show()

mean_x = np.mean(measure_x)
std_dev_x = np.std(measure_x)
variance_x = np.var(measure_x)

print("Mean (μ): {:.2f}%".format(mean_x))
print("Standard Deviation (σ): {:.2f}%".format(std_dev_x))
print("Variance (σ^2): {:.2f}%".format(variance_x))