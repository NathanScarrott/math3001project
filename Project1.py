import numpy as np
import matplotlib.pyplot as plt

# Convert all lists to numpy arrays
num_people_per_bin = np.array([3100000.0, 6450000.0, 9220000.0, 5110000.0, 3030000.0, 
                               2410000.0, 1140000.0, 547000.0, 183000.0, 131000.0, 
                               74000.0, 37000.0, 19000.0])
income_per_bin = np.array([42600000000.0, 113000000000.0, 226000000000.0, 177000000000.0, 
                           135000000000.0, 139000000000.0, 94200000000.0, 65300000000.0, 
                           31400000000.0, 31400000000.0, 28000000000.0, 25400000000.0, 
                           47300000000.0])
mean_salary_given = np.array([13742, 17519, 24512, 34638, 44554, 57676, 82632, 119378, 
                              171585, 239695, 378378, 686486, 2489474])

# UK tax brackets
tb1, tb2, tb3, tb4 = [12570, 0], [50270, 0.2], [125140, 0.4], [125140, 0.45] 

# Calculate the mean salary per bin (no tax)
mean_salary_calculated = income_per_bin / num_people_per_bin

# Total income
I = income_per_bin.sum()

# Cumulative sum of income per bin
cum_income = np.cumsum(income_per_bin) / I

# Function to calculate tax per bin given the salary
def calculate_tax(salary, tb1, tb2, tb3, tb4):
    if salary <= tb1[0]:
        return salary * tb1[1]
    elif salary <= tb2[0]:
        return (tb1[0] * tb1[1]) + ((salary - tb1[0]) * tb2[1])
    elif salary <= tb3[0]:
        return (tb1[0] * tb1[1]) + ((tb2[0] - tb1[0]) * tb2[1]) + ((salary - tb2[0]) * tb3[1])
    else:
        return (tb1[0] * tb1[1]) + ((tb2[0] - tb1[0]) * tb2[1]) + ((tb3[0] - tb2[0]) * tb3[1]) + ((salary - tb3[0]) * tb4[1])

# Calculate the tax for each mean salary calculated
tax_per_bin = np.array([calculate_tax(salary, tb1, tb2, tb3, tb4) for salary in mean_salary_calculated])

# Calculate the total tax for each bin by multiplying the tax per person by the number of people
total_tax_per_bin = tax_per_bin * num_people_per_bin

# Adjusted income after tax for each bin
income_adjusted_tax = income_per_bin - total_tax_per_bin


# Calculate cumulative people as a fraction of the total number of people
total_people = num_people_per_bin.sum()
cum_people = np.cumsum(num_people_per_bin) / total_people




# Calculate cumulative income after tax as a fraction of the total adjusted income
cum_income_after_tax = np.cumsum(income_adjusted_tax) / income_adjusted_tax.sum()

# Prepare the data for plotting
cum_people_with_zero = np.insert(cum_people, 0, 0)
cum_income_with_zero = np.insert(cum_income, 0, 0)
cum_income_after_tax_with_zero = np.insert(cum_income_after_tax, 0, 0)








N = sum(num_people_per_bin)
cum_amount_population = np.cumsum(num_people_per_bin)/N
print(cum_amount_population)
print(cum_income)

 
cum_amount_population_with_zero = np.insert(cum_amount_population, 0, 0)
cum_income_with_zero = np.insert(cum_income, 0, 0)
cum_income_after_tax_with_zero = np.insert(cum_income_after_tax, 0, 0)

# Adjusting the subplot to ensure the titles and labels don't clash
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
# Add a 45% equality line, which represents perfect equality in income distribution
equality_line = np.linspace(0, 1, len(cum_people_with_zero))

# Plot cumulative income before tax
axs[0].plot(cum_amount_population_with_zero, cum_income_with_zero, marker='o', label='Cumulative Income Before Tax')
axs[0].plot(equality_line, equality_line, 'r--', label='Equality Line')
axs[0].set_xlabel('Cumulative People (as a fraction of total people)')
axs[0].set_ylabel('Cumulative Income (as a fraction of total income)')
axs[0].set_title('Cumulative Income Before Tax vs Cumulative People')
axs[0].grid(True)
axs[0].legend()

# Adjusting the layout to prevent title and x-axis label overlap
plt.setp(axs[0], xticks=np.arange(0, 1.1, 0.1), yticks=np.arange(0, 1.1, 0.1))
plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[0].yaxis.get_majorticklabels(), rotation=45)

# Plot cumulative income after tax
axs[1].plot(cum_amount_population_with_zero, cum_income_after_tax_with_zero, marker='o', label='Cumulative Income After Tax')
axs[1].plot(equality_line, equality_line, 'r--', label='Equality Line')
axs[1].set_xlabel('Cumulative People (as a fraction of total people)')
axs[1].set_ylabel('Cumulative Income After Tax (as a fraction of total adjusted income)')
axs[1].set_title('Cumulative Income After Tax vs Cumulative People with Equality Line and Point at (0,0)')
axs[1].grid(True)
axs[1].legend()

# Adjusting the layout to prevent title and x-axis label overlap
plt.setp(axs[1], xticks=np.arange(0, 1.1, 0.1), yticks=np.arange(0, 1.1, 0.1))
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[1].yaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
#plt.show()



plt.figure(15) 
# plt.subplot(2,2,4)
Ns = 500
ss = np.linspace(0,2*10**5, Ns)
piss = np.copy(ss)
tb1 = 12570
tb2 = 50270
tb3 = 150000
db1 = 0.2
db2 = 0.2
db3 = 0.05
piss = db1*np.heaviside(ss-tb1,0.5)+db2*np.heaviside(ss-tb2,0.5)+db3*np.heaviside(ss-tb3,0.5)
plt.plot(ss,piss,'-b', linewidth=2)
plt.xlabel(f"$s$",fontsize=16)
plt.ylabel(f"$\pi(s)$ ",fontsize=16)
#plt.show()

# We will calculate the cumulative sum of the tax rate function within each tax bracket.

# Initialize the cumulative sum array
cumsum_piss = np.zeros_like(piss)

# Calculate the cumulative sum manually for each tax bracket
for i in range(1, Ns):
    # Sum the tax rate up to the current salary `ss[i]`
    cumsum_piss[i] = cumsum_piss[i-1] + piss[i] * (ss[i] - ss[i-1])

# Calculate the cumulative sum at the tax bracket thresholds
cumsum_tb1 = cumsum_piss[ss.searchsorted(tb1)]
cumsum_tb2 = cumsum_piss[ss.searchsorted(tb2)]
cumsum_tb3 = cumsum_piss[ss.searchsorted(tb3)]

cumsum_tb1, cumsum_tb2, cumsum_tb3, cumsum_piss[-1]

print(cumsum_tb1)
print(cumsum_tb2)
print(cumsum_tb3)

# We will plot the cumulative sum of the tax rate function (piss) on top of the previous Lorenz curve plot.
# First, we need to normalize the cumulative sum to the range [0, 1] to match the Lorenz curve scale.

# Normalize the cumulative sum by dividing by the last value (total area under the curve)
normalized_cumsum_piss = cumsum_piss / cumsum_piss[-1]

# Plotting cumulative income before and after tax on the same graph with the cumulative tax rate curve
plt.figure(figsize=(10, 6))

# Plot cumulative income before tax
plt.plot(cum_amount_population_with_zero, cum_income_with_zero, marker='o', label='Cumulative Income Before Tax')

# Plot cumulative income after tax
plt.plot(cum_amount_population_with_zero, cum_income_after_tax_with_zero, marker='o', label='Cumulative Income After Tax')

# Plot the normalized cumulative tax rate function
plt.plot(ss / max(ss), normalized_cumsum_piss, '-g', label='Cumulative Tax Rate Function')

# Add a 45% equality line, which represents perfect equality in income distribution
plt.plot(equality_line, equality_line, 'r--', label='Equality Line')

# Adding labels, legend, and title
plt.xlabel('Cumulative People (as a fraction of total people)')
plt.ylabel('Cumulative Income / Cumulative Tax Rate')
plt.title('Lorenz Curve with Cumulative Tax Rate Function')
plt.legend()
plt.grid(True)
plt.show()


# Assuming 'cum_people_with_zero' and 'cum_income_with_zero' are your cumulative population and income
# Calculate the area under the Lorenz curve using the trapezoidal rule
area_under_lorenz_curve = np.trapz(cum_income_with_zero, cum_people_with_zero)

# The area under the line of equality is 0.5 (the area of a triangle with base and height of 1)
area_under_line_of_equality = 0.5

# The Gini coefficient is equal to the area between the line of equality and the Lorenz curve,
# normalized by the total area under the line of equality (which is 0.5).
gini_coefficient = (area_under_line_of_equality - area_under_lorenz_curve) / area_under_line_of_equality

print(gini_coefficient)

# Define the function lambda(x)
lambda_x = (1 - a) * x**b + a * (1 - (1 - x)**(1/b))

# Display the function
lambda_x
