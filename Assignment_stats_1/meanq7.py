# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:10:59 2023

@author: Masum
"""

import pandas as pd
data = pd.read_csv("C:\\Users\\Masum\\Downloads\\Q7.csv")

# Calculate mean for each variable
mean_points = data['Points'].mean()
mean_score = data['Score'].mean()
mean_weigh = data['Weigh'].mean()

# Calculate median for each variable
median_points = data['Points'].median()
median_score = data['Score'].median()
median_weigh = data['Weigh'].median()

# Calculate mode for each variable
mode_points = data['Points'].mode()
mode_score = data['Score'].mode()
mode_weigh = data['Weigh'].mode()

# Calculate variance for each variable
variance_points = data['Points'].var()
variance_score = data['Score'].var()
variance_weigh = data['Weigh'].var()

# Calculate standard deviation for each variable
standard_deviation_points = data['Points'].std()
standard_deviation_score = data['Score'].std()
standard_deviation_weigh = data['Weigh'].std()

# Calculate range for each variable
range_points = data['Points'].max() - data['Points'].min()
range_score = data['Score'].max() - data['Score'].min()
range_weigh = data['Weigh'].max() - data['Weigh'].min()

# Print the results
print("Mean of Points:", mean_points)
print("Mean of Score:", mean_score)
print("Mean of Weigh:", mean_weigh)

print("Median of Points:", median_points)
print("Median of Score:", median_score)
print("Median of Weigh:", median_weigh)

print("Mode of Points:", mode_points)
print("Mode of Score:", mode_score)
print("Mode of Weigh:", mode_weigh)

print("Variance of Points:", variance_points)
print("Variance of Score:", variance_score)
print("Variance of Weigh:", variance_weigh)

print("Standard Deviation of Points:", standard_deviation_points)
print("Standard Deviation of Score:", standard_deviation_score)
print("Standard Deviation of Weigh:", standard_deviation_weigh)

print("Range of Points:", range_points)
print("Range of Score:", range_score)
print("Range of Weigh:", range_weigh)