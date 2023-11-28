# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:11:35 2023

@author: Masum
"""

import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# Read data from the external file (assumes a CSV file)
file_path = "C:\\Users\\Masum\\Downloads\\Q9_a.csv"
data = pd.read_csv(file_path)

speed_data = data['speed']
distance_data = data['dist']
 # Calculate skewness and kurtosis
speed_skew = skew(speed_data)
distance_skew = skew(distance_data)
speed_kurt = kurtosis(speed_data)
distance_kurt = kurtosis(distance_data)
   # Print the results
print(f"Speed Skewness: {speed_skew}")
print(f"Distance Skewness: {distance_skew}")
print(f"Speed Kurtosis: {speed_kurt}")
print(f"Distance Kurtosis: {distance_kurt}")
   # Draw histograms for visual inspection
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.hist(speed_data, bins=20, color='blue', alpha=0.7)
plt.title('Speed Distribution')

plt.subplot(2, 2, 2)
plt.hist(distance_data, bins=20, color='green', alpha=0.7)
plt.title('Distance Distribution')

plt.subplot(2, 2, 3)
plt.boxplot(speed_data)
plt.title('Speed Boxplot')

plt.subplot(2, 2, 4)
plt.boxplot(distance_data)
plt.title('Distance Boxplot')

plt.tight_layout()
plt.show()

