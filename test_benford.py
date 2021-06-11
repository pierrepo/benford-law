"""Test use of the benford module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import benford as ben


def test_get_theoretical_freq_benford():
    """
    Test if theoretical proportion of the first digits considered is
    correct.
    """
    # Setup
    correct_freq_base10_1digit = np.array([0.30103,    0.17609126, 0.12493874,
                                           0.09691001, 0.07918125, 0.06694679,
                                           0.05799195, 0.05115252, 0.0457574])

    correct_freq_base10_2digit = np.array([0.04139269, 0.03778856, 0.03476211,
                                           0.03218468, 0.02996322, 0.02802872,
                                           0.02632894, 0.02482358, 0.0234811,
                                           0.02227639, 0.0211893,  0.02020339,
                                           0.01930516, 0.01848341, 0.01772877,
                                           0.01703334, 0.01639042, 0.01579427,
                                           0.01523997, 0.01472326, 0.01424044,
                                           0.01378828, 0.01336396, 0.01296498,
                                           0.01258913, 0.01223446, 0.01189922,
                                           0.01158187, 0.01128101, 0.01099538,
                                           0.01072387, 0.01046543, 0.01021917,
                                           0.00998422, 0.00975984, 0.00954532,
                                           0.00934003, 0.00914338, 0.00895484,
                                           0.00877392, 0.00860017, 0.00843317,
                                           0.00827253, 0.00811789, 0.00796893,
                                           0.00782534, 0.00768683, 0.00755314,
                                           0.00742402, 0.00729924, 0.00717858,
                                           0.00706185, 0.00694886, 0.00683942,
                                           0.00673338, 0.00663058, 0.00653087,
                                           0.00643411, 0.00634018, 0.00624895,
                                           0.00616031, 0.00607415, 0.00599036,
                                           0.00590886, 0.00582954, 0.00575233,
                                           0.00567713, 0.00560388, 0.00553249,
                                           0.0054629,  0.00539503, 0.00532883,
                                           0.00526424, 0.00520119, 0.00513964,
                                           0.00507953, 0.0050208,  0.00496342,
                                           0.00490733, 0.0048525,  0.00479888,
                                           0.00474644, 0.00469512, 0.00464491,
                                           0.00459575, 0.00454763, 0.0045005,
                                           0.00445434, 0.00440912, 0.0043648])

    correct_freq_base8_1digit = np.array([0.33333333, 0.1949875,  0.13834583,
                                          0.10730936, 0.08767814, 0.07413081,
                                          0.06421503])

    correct_freq_base8_2digit = np.array([0.05664167, 0.0506677,  0.04583451,
                                          0.04184363, 0.03849241, 0.0356384,
                                          0.03317856, 0.03103647, 0.02915428,
                                          0.02748739, 0.02600084, 0.02466686,
                                          0.02346311, 0.0223714,  0.02137678,
                                          0.02046685, 0.01963123, 0.01886118,
                                          0.01814926, 0.01748914, 0.01687536,
                                          0.0163032,  0.01576857, 0.0152679,
                                          0.01479804, 0.01435624, 0.01394006,
                                          0.01354733, 0.01317612, 0.01282472,
                                          0.01249157, 0.01217529, 0.01187464,
                                          0.01158847, 0.01131578, 0.01105562,
                                          0.01080716, 0.01056962, 0.0103423,
                                          0.01012455, 0.00991578, 0.00971545,
                                          0.00952305, 0.00933813, 0.00916025,
                                          0.00898902, 0.00882407, 0.00866507,
                                          0.0085117,  0.00836366, 0.00822068,
                                          0.00808252, 0.00794891, 0.00781966,
                                          0.00769454, 0.00757336])

    # Exercise
    current_freq_base10_1digit = ben.get_theoretical_freq_benford()
    current_freq_base10_2digit = ben.get_theoretical_freq_benford(nb_digit=2)
    current_freq_base8_1digit = ben.get_theoretical_freq_benford(nb_digit=1,
                                                                 base=8)
    current_freq_base8_2digit = ben.get_theoretical_freq_benford(nb_digit=2,
                                                                 base=8)

    # Verify
    assert_array_almost_equal(correct_freq_base10_1digit,
                              current_freq_base10_1digit, 5)
    assert_array_almost_equal(correct_freq_base10_2digit,
                              current_freq_base10_2digit, 5)
    assert_array_almost_equal(correct_freq_base8_1digit,
                              current_freq_base8_1digit, 5)
    assert_array_almost_equal(correct_freq_base8_2digit,
                              current_freq_base8_2digit, 5)

    # Cleanup - None


def test_count_first_digit():
    """
    Test if distribution of the first significant digits of observed
    data is correct.
    """
    # Setup
    correct_first_digit = np.array([1, 2, 1, 2, 0, 1, 2, 1, 0])
    correct_2first_digit = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                     1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                     1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Exercise
    numbers = [12, 458, 846, 7845, 25, 65, 48, 708, 201, 35]
    current_first_digit_positf = ben.count_first_digit(numbers, nb_digit=1)

    numbers = [-12, -458, -846, -7845, -25, -65, -48, -708, -21, -35]
    current_first_digit_negatif = ben.count_first_digit(numbers, nb_digit=1)

    numbers = [0.0000012, 0.458, 8.46, 78.45, 0.025, 6.5, 0.0000048, 0.07080,
               2.01, 0.0000000000000035]
    current_first_digit_positif_float = ben.count_first_digit(numbers,
                                                              nb_digit=1)

    numbers = [-0.0000012, -0.458, -8.46, -78.45, -0.025, -6.5, -0.0000048,
               -0.07080, -2.01, -0.0000000000000035]
    current_first_digit_negaitf_float = ben.count_first_digit(numbers,
                                                              nb_digit=1)

    numbers = [12, 458, 846, 7845, 25, 65, 48, 708, 201, 35]
    current_2irst_digit_positf = ben.count_first_digit(numbers, nb_digit=2)

    numbers = [-12, -458, -846, -7845, -25, -65, -48, -708, -201, -35]
    current_2irst_digit_negatif = ben.count_first_digit(numbers, nb_digit=2)

    numbers = [0.0000012, 0.458, 8.46, 78.45, 0.025, 6.5, 0.0000048, 0.07080,
               2.01, 0.0000000000000035]
    current_2irst_digit_positif_float = ben.count_first_digit(numbers,
                                                              nb_digit=2)

    numbers = [-0.0000012, -0.458, -8.46, -78.45, -0.025, -6.5, -0.0000048,
               -0.07080, -2.01, -0.0000000000000035]
    current_2irst_digit_negaitf_float = ben.count_first_digit(numbers,
                                                              nb_digit=2)

    # Verify
    assert_array_almost_equal(current_first_digit_positf,
                              correct_first_digit)
    assert_array_almost_equal(current_first_digit_negatif,
                              correct_first_digit)
    assert_array_almost_equal(current_first_digit_positif_float,
                              correct_first_digit)
    assert_array_almost_equal(current_first_digit_negaitf_float,
                              correct_first_digit)
    assert_array_almost_equal(current_2irst_digit_positf,
                              correct_2first_digit)
    assert_array_almost_equal(current_2irst_digit_negatif,
                              correct_2first_digit)
    assert_array_almost_equal(current_2irst_digit_positif_float,
                              correct_2first_digit)
    assert_array_almost_equal(current_2irst_digit_negaitf_float,
                              correct_2first_digit)

    # Cleanup - None


def test_normalize_first_digit():
    """
    Test if Normalize observed distribution of the first significant
    digits is correct.
    """
    # Setup
    correct_norm_first_digit = [0.02,  0.01,  0.05,  0.13,  0.14, 0.138,
                                0.156, 0.098, 0.124, 0.092, 0.042]

    # Exercise
    numbers = np.array([10, 5, 25, 65, 70, 69, 78, 49, 62, 46, 21])
    current_norm_first_digit = ben.normalize_first_digit(numbers)

    # Verify
    assert_array_almost_equal(correct_norm_first_digit,
                              current_norm_first_digit)

    # Cleanup - None


@pytest.mark.mlp_image_compare()
def test_build_hist_freq_ben():
    """
    Test if Histogram of observed proportion and theoretical
    proportion is correct.
    """
    # Setup
    freq_obs = np.array([0.30103,    0.17609126, 0.12493874,
                         0.09691001, 0.07918125, 0.06694679,
                         0.05799195, 0.05115252, 0.0457574])
    freq_theo = np.array([0.30, 0.17, 0.12,
                          0.09, 0.07, 0.06,
                          0.05, 0.05, 0.04])

    # Exercise
    hist = ben.build_hist_freq_ben(freq_obs, freq_theo, nb_digit=1)

    # Verify - Done elsewhere

    # Cleanup - None

    return hist


def test_calculate_pom():
    """
    Test is physical order of magnitude is correct.
    """
    # Setup
    correct_pom = 47016.3806552262

    # Exercise
    data_obs = np.array([0.52, 12, 12055, 548, 275, 23.215, 0.2564])
    current_pom = ben.calculate_pom(data_obs)

    # Verify
    assert_almost_equal(correct_pom, current_pom, 5)

    # Cleanup - None


if __name__ == "__main__":
    print("\nThis is test script for benford module.\n"
          "Enter : pytest --cov-report term-missing --cov\n"
          "To test the benford module\n")
