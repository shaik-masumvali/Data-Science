
Cars <- read.csv("C:/Users/Masum/Downloads/cars.csv")

# Extract the MPG column
MPG <- Cars$MPG

# a. P(MPG > 38)
prob_a <- length(MPG[MPG > 38]) / length(MPG)

# b. P(MPG < 40)
prob_b <- length(MPG[MPG < 40]) / length(MPG)

# c. P(20 < MPG < 50)
prob_c <- length(MPG[MPG > 20 & MPG < 50]) / length(MPG)

# Display the probabilities
cat("a. P(MPG > 38):", prob_a, "\n")
cat("b. P(MPG < 40):", prob_b, "\n")
cat("c. P(20 < MPG < 50):", prob_c, "\n")


# Shapiro-Wilk Test for Normality
shapiro_test_result <- shapiro.test(MPG)

# Display the result of the normality test
cat("Shapiro-Wilk Test for Normality:\n")
cat("p-value:", shapiro_test_result$p.value, "\n")

# Check the null hypothesis
if (shapiro_test_result$p.value > 0.05) {
  cat("The MPG data appears to be normally distributed (fail to reject the null hypothesis).\n")
} else {
  cat("The MPG data does not appear to be normally distributed (reject the null hypothesis).\n")
}

