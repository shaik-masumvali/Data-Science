# Calculate the t-score
t_score = (260 - 270) / (90 / sqrt(18))

# Calculate the degrees of freedom
df = 18 - 1

# Calculate the probability
p = pt(t_score, df)

# Print the probability
print(p)
