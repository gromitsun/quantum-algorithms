import numpy as np

from qiskit.aqua.algorithms.amplitude_estimators.ae_utils import bisect_max, pdf_a


def mle(qae: float, ai: np.array, pi: np.array, m: int, shots: int):
    """
    Compute the Maximum Likelihood Estimator (MLE).
    :param qae: Estimated a value from AE with maximum probability
    :param ai: All measured values of a
    :param pi: Probabilities / counts measured for the values in ai
    :param m: Number of evaluation qubits
    :param shots: Number of shots taken
    :return: The MLE for the AE run
    """
    def loglikelihood(a):
        return np.sum(shots * pi * np.log(pdf_a(ai, a, m)))

    mm = 2**m

    # y is pretty much an integer, but to map 1.9999 to 2 we must first
    # use round and then int conversion
    y = int(np.round(mm * np.arcsin(np.sqrt(qae)) / np.pi))

    # Compute the two intervals in which are candidates for containing
    # the maximum of the log-likelihood function: the two bubbles next to
    # the QAE estimate
    if y == 0:
        right_of_qae = np.sin(np.pi * (y + 1) / mm)**2
        bubbles = [qae, right_of_qae]

    elif y == int(mm / 2):  # remember, M = 2^m is a power of 2
        left_of_qae = np.sin(np.pi * (y - 1) / mm)**2
        bubbles = [left_of_qae, qae]

    else:
        left_of_qae = np.sin(np.pi * (y - 1) / mm)**2
        right_of_qae = np.sin(np.pi * (y + 1) / mm)**2
        bubbles = [left_of_qae, qae, right_of_qae]

    # Find global maximum amongst the two local maxima
    a_opt = qae
    loglik_opt = loglikelihood(a_opt)
    for a, b in zip(bubbles[:-1], bubbles[1:]):
        locmax, val = bisect_max(loglikelihood, a, b, retval=True)
        if val > loglik_opt:
            a_opt = locmax
            loglik_opt = val

    return a_opt
