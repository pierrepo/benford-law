"""Module to verify Benford's law on observed data."""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import distributions, power_divergence


def get_theoretical_freq_benford(nb_digit=1):
    """Theoretical proportions of Benford's law.

    Function to return the theoretical proportion of the first
    significant digits.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    nb_digit : int
        Number of first digits to consider.

    Returns
    ¯¯¯¯¯¯¯
    p_benford : array
        Theoretical proportion of the first digits considered.

    """
    digit = (10 ** nb_digit) - (10 ** (nb_digit - 1))
    p_benford = np.zeros(digit, dtype=float)
    for i in range(digit):
        p_benford[i] = (math.log((1 + (1 / (i + (10 ** (nb_digit - 1))))),
                                 10))
    return p_benford


def count_first_digit(numbers, nb_digit=1):
    """Distribution of the first significant digits of observed data.

    Function to return the observed distribution of the first digits of
    an observed data set.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    numbers : array of numbers
        Integer array.
    nb_digit : int
        Number of first significant digits.

    Returns
    ¯¯¯¯¯¯¯
    digit_distrib : array
        Distribution of the first significant digits.

    """
    digit = (10 ** nb_digit) - (10 ** (nb_digit - 1))
    # array size return
    digit_distrib = np.zeros(digit, dtype=int)
    for number in numbers:
        if len(str(number)) > nb_digit:
            while str(number)[1] == ".":
                number *= 10 ** nb_digit
        first = int(str(number)[0:nb_digit])
        digit_distrib[first - (10 ** (nb_digit - 1))] += 1
    return digit_distrib


def normalize_first_digit(array):
    """Normalize observed distribution of the first significant digits.

    Function normalizing an array by the sum of the array values.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    array: array of int
        Array of observed data.

    Returns
    ¯¯¯¯¯¯¯
    array: array of float
        Array of observed data normalized.

    """
    array = array / sum(array)
    return array


def build_hist_freq_ben(freq_obs, freq_theo, nb_digit, title="",
                        xlab="First digit", ylab="Proportion",
                        legend="", name_save="", size=(6, 4)):
    """Histogram of observed proportion and theoretical proportion.

    Function realizing the histogram of observed proportions and adding
    the theoretical proportion of Benford.

    Parameters
    ¯¯¯¯¯¯¯¯¯¯
    freq_obs : array
        Array of observed frequency.
    freq_theo : array
        Array of theoritical frequency.
    nb_digit : int
        Number of first significant digits.
    title : string, optinal
        Title of histogram.
    xlab: string, optinal
        Label of x-axis. Default is `"First digit"`.
    ylab: string, optional
        Label of y-axis. Default is `"Proportion"`.
    legend: string, optional
        Label of the legend for the theoretical frequency.
    name_save: string, optional
        Name of the image to save in .png format, if you want to save it.
    size: tuple of 2 int, optional
        Plot size. Default is `(6, 4)`.

    Returns
    ¯¯¯¯¯¯¯
    Histogram.

    """
    plt.figure(figsize=size)
    plt.plot(range(1, len(freq_theo)+1), freq_theo, marker="o",
             color="red")
    plt.bar(range(1, len(freq_obs)+1), freq_obs)

    lab = []
    for i in range((10 ** (nb_digit-1)), (10 ** nb_digit)):
        lab.append(str(i))

    plt.xticks(ticks=range(1, len(freq_theo)+1), labels=lab)
    plt.title(label=title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(labels=("Benford's law", legend))
    if name_save != "":
        plt.savefig(f"{name_save}.png", transparent=True)


def calculate_bootstrap_chi2(data_obs, f_theo, nb_digit, nb_val=1000,
                             nb_loop=1000, type_test=1):
    """Average of calculated chi2 and asociate p_value.

    Function to calculate average chi2 in the function bootstrap_chi2.

    parameters
    ¯¯¯¯¯¯¯¯¯¯
    data_obs : array of int
        Integer array of observed dataset.
    f_theo : array of float
        Float array of theoretical frequency.
    nb_digit: int
        Number of first significant digits. Default is `1`.
    nb_val : int, optinal
        Sample size. Default is `1000`.
    nb_loop : int, optional
        number of "bootstrap" procedure is performed. Default is `1000`.
    type_test: string or int, optional
        statistical test type performed. Default is `1`.
            String            Value   test type
            "pearson"           1     Chisquare-test.
            "log-likelihood"    0     G-test.

    Returns
    ¯¯¯¯¯¯¯
    mean_chi2: float
        Chi2 average of "bootstrap".
    p_val
        p-value of mean_chi2.
    nb_signif: int
        number of significant statistical tests in the "bootstrap"

    """
    sum_chi2 = np.zeros(nb_loop, dtype=float)
    d_theo = np.array(f_theo * nb_val)
    nb_signif = 0
    for i in range(nb_loop):
        ech = np.random.choice(data_obs, size=nb_val, replace=False)
        d_obs = count_first_digit(ech, nb_digit)
        sum_chi2[i], p_v = power_divergence(f_obs=d_obs, f_exp=d_theo,
                                            lambda_=type_test)
        if p_v < 0.05:
            nb_signif += 1
    mean_chi2 = sum(sum_chi2) / nb_loop
    k = len(f_theo+1)
    p_val = distributions.chi2.sf(mean_chi2, k - 1)
    print(f"statistics : {mean_chi2} ; p-value : {p_val} ; "
          f"number of significant tests : {nb_signif}")
    return mean_chi2, p_val, nb_signif


if __name__ == "__main__":
    print("\nThis is benford module. This module contains functions to "
          "analyze a data set according to Benford's law.\n")
