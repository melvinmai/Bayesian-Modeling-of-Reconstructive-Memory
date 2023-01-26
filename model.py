"""
Melvin Mai
CSC292
Dr Robert Jacobs
22 November 2022
Problem 2
"""

import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt 

# Object Mean
obj_mean = np.array([0.2, 0.4, 0.6, 0.8])
obj_var = 0.005

# Category Mean Variance
cat_mean = 0.5
cat_var = 0.5

# Z, bernoulli distribution
z = np.random.binomial(1, 0.7, 1)

# Noise Var
noise = 0.02

# This is a single trial 
# Returns the memory error for each possible absolute study size [0.1, 0.9]
def sim(n, obj_mean, obj_var, cat_mean, cat_var, fam):
    # Generate familiarity parameter z = 1, 0 based on familiarity 0
    z = np.random.binomial(1, p=fam)

    # Prior mean and variances weighted by familiarity z
    # Use object info if familiar z = 1, else, use category info
    prior_mean = z * obj_mean + (1 - z) * cat_mean
    prior_variance = z * obj_var + (1 - z) * cat_var

    # Generate absolute study size
    possible_absolute_mean = np.linspace(0.1, 0.9, 9, endpoint=True)

    # Generate mean of study size
    list_mem_error = []
    for s in possible_absolute_mean:
        y_hat = []
        mu_s = np.random.normal(s, np.sqrt(0.005), 1)

        for j in range(n):
            # Generate Episodic memories
            y_i = np.random.normal(mu_s, np.sqrt(noise), 1)
            y_hat.append(y_i)

        # Mean of episodic Memory
        y_hat = np.mean(y_hat)

        # Posterior Mean
        w = (1/prior_variance)/((1/prior_variance)+(n/noise))
        mu_n = w * prior_mean + (1 - w) * y_hat

        mem_error = mu_n - mu_s

        list_mem_error.append(mem_error)

    return np.array(list_mem_error)

# This function runs the simulation n number of times
# This function averages all simulation data
# Returns an array of averaged data points for each absolute size [0.1, 0.9]
def run_sim(runs, mean, fam, n):
    sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9 = 0,0,0,0,0,0,0,0,0
    for i in range(runs):
        new_data = sim(n, mean, obj_var, cat_mean, cat_var, fam)
        sum1 += new_data[0]
        sum2 += new_data[1]
        sum3 += new_data[2]
        sum4 += new_data[3]
        sum5 += new_data[4]
        sum6 += new_data[5]
        sum7 += new_data[6]
        sum8 += new_data[7]
        sum9 += new_data[8]

    sum = [sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9]
    avg_mem_error = np.asarray(sum) / runs
    return avg_mem_error

def main():
    # Familiarity, n is number of memory samples, runs is number of simulation trials
    # familiarity can either be 0.0, 0.4, 0.7
    # n can either be 4 or 40 for number of memory samples
    familiarity = 0.0
    n = 40
    runs = 1000

    # Create an array of data for each object mean
    small = run_sim(runs, 0.2, familiarity, n)
    med_small = run_sim(runs, 0.4, familiarity, n)
    med_large = run_sim(runs, 0.6, familiarity, n)
    large = run_sim(runs, 0.8, familiarity, n)


    # Plot
    fig = plt.figure(figsize=(7, 5))
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    x_range = np.linspace(0.1, 0.9, 9, endpoint=True)
    axes.plot(x_range, small, 'red', label="Mean: 0.2")
    axes.plot(x_range, med_small, 'blue', label="Mean: 0.4")
    axes.plot(x_range, med_large, 'green', label="Mean: 0.6")
    axes.plot(x_range, large, 'purple', label="Mean: 0.8")
    axes.set_xlabel('Possible Absolute Size')
    axes.set_ylabel('Memory Error')
    axes.legend(loc=0)
    plt.title("Familiarity = 0.7")
    plt.ylim([-0.3, 0.3])
    plt.show()

main()