
# Load the data set
wc_at <- read.csv("C:\Users\Masum\Downloads\wc-at.csv")

# Calculate the skewness and kurtosis of AT
at_skew <- skewness(wc_at$AT)
at_kurt <- kurtosis(wc_at$AT)

# Calculate the skewness and kurtosis of Waist
waist_skew <- skewness(wc_at$Waist)
waist_kurt <- kurtosis(wc_at$Waist)

# Print the results
print("Skewness of AT:", at_skew)
print("Kurtosis of AT:", at_kurt)

print("Skewness of Waist:", waist_skew)
print("Kurtosis of Waist:", waist_kurt)