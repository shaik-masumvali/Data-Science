# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:35:26 2023

@author: Masum
"""


import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# Read data from the external file (assumes a CSV file)
file_path = "C:\\Users\\Masum\\Downloads\\Q9_b.csv"
data = pd.read_csv(file_path)
# Extract columns
sp_column = data['SP']
wt_column = data['WT']

# Calculate skewness and kurtosis
sp_skewness = skew(sp_column)
wt_skewness = skew(wt_column)

sp_kurtosis = kurtosis(sp_column)
wt_kurtosis = kurtosis(wt_column)

# Print results
print(f"Skewness for SP: {sp_skewness}")
print(f"Skewness for WT: {wt_skewness}")
print(f"Kurtosis for SP: {sp_kurtosis}")
print(f"Kurtosis for WT: {wt_kurtosis}")

# Draw inferences based on the results
if sp_skewness > 0:
    print("The SP data is positively skewed.")
elif sp_skewness < 0:
    print("The SP data is negatively skewed.")
else:
    print("The SP data is approximately symmetric.")

if wt_skewness > 0:
    print("The WT data is positively skewed.")
elif wt_skewness < 0:
    print("The WT data is negatively skewed.")
else:
    print("The WT data is approximately symmetric.")

if sp_kurtosis > 0:
    print("The SP data is leptokurtic.")
elif sp_kurtosis < 0:
    print("The SP data is platykurtic.")
else:
    print("The SP data has a normal distribution of kurtosis.")

if wt_kurtosis > 0:
    print("The WT data is leptokurtic.")
elif wt_kurtosis < 0:
    print("The WT data is platykurtic.")
else:
    print("The WT data has a normal distribution of kurtosis.")

