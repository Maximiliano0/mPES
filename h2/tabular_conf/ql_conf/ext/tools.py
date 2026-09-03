'''
pes_ql - Pandemic Experiment Scenario: Utility Functions

Small collection of helper functions used across pes_ql modules:

- **entropy_from_pdf**:  Shannon entropy (bits) of a probability distribution.
- **convert_globalseq_to_seqs**:  Splits a flat values array into nested
  lists grouped by per-sequence lengths.
- **plot_confidences**:  Histogram visualization of confidence scores.
'''

##########################
##  Imports externos    ##
##########################
import numpy
import matplotlib.pyplot as plt



##########################
##  Imports internos    ##
##########################


def entropy_from_pdf(pdf):
    '''
    Return the entropy of the provided pdf (which can be a histogram).

    Parameters
    ----------
    pdf : array-like
        Probability distribution function values (can be a histogram)

    Returns
    -------
    float
        Shannon entropy of the distribution in bits
    '''
    # Push all the values upwards for them to be all positive.
    pdf = pdf + numpy.abs(numpy.min(pdf))

    # Normalize them to be numpy.sum(pdf) = 1 (probability)
    p = pdf / numpy.sum(pdf)
    p[p == 0] += 0.000001  # Avoid zero value, by adding just a small epsilon
    H = -numpy.dot(p, numpy.log2(p))
    return H


def confidence_from_q_values(options, resources_left):
    '''Return entropy-based confidence for the feasible Q-values only.

    Parameters
    ----------
    options : array-like
        Q-values for every action in the current state.
    resources_left : int
        Number of resources remaining, which defines the feasible actions.

    Returns
    -------
    float
        Normalized confidence in the interval ``[0.0, 1.0]``. A single
        feasible action has confidence ``1.0``; tied feasible actions have
        confidence ``0.0``.
    '''
    q_values = numpy.asarray(options, dtype=numpy.float64)
    maximum_action = min(max(int(resources_left), 0), len(q_values) - 1)
    feasible_q_values = q_values[:maximum_action + 1]

    if len(feasible_q_values) == 1:
        return 1.0
    if numpy.allclose(feasible_q_values, feasible_q_values[0]):
        return 0.0

    min_entropy_distribution = numpy.zeros(len(feasible_q_values))
    min_entropy_distribution[0] = 1
    max_entropy_distribution = numpy.ones(len(feasible_q_values))
    decision_entropy = entropy_from_pdf(feasible_q_values)
    min_entropy = entropy_from_pdf(min_entropy_distribution)
    max_entropy = entropy_from_pdf(max_entropy_distribution)
    confidence = (decision_entropy - max_entropy) / (min_entropy - max_entropy)
    return float(numpy.clip(confidence, 0.0, 1.0))


def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    '''Return the greedy feasible action, confidence, and simulated response times.

    Parameters
    ----------
    options : array-like
        Q-values for the available actions.
    resources_left : int
        Number of resources remaining.
    response_timeout : float
        Maximum response time in milliseconds.

    Returns
    -------
    tuple
        Selected action, normalized confidence, hold time, and release time.
    '''
    confidence = confidence_from_q_values(options, resources_left)
    options = numpy.asarray(options, dtype=numpy.float64).copy()
    feasible_actions = numpy.arange(len(options), dtype=numpy.float32)
    options[feasible_actions > resources_left] = -1e9

    response = int(numpy.argmax(options))
    response_time_mean = int((1 - 2 * confidence) * 10)
    hold_time = numpy.random.normal(response_time_mean, 3)
    release_time = hold_time + numpy.random.normal(response_time_mean, 1)
    maximum_time = response_timeout / 1000.0

    return (
        response,
        confidence,
        numpy.clip(hold_time, 0, maximum_time),
        numpy.clip(release_time, 0, maximum_time),
    )


def convert_globalseq_to_seqs(sequence_map, seqin360):
    '''
    Convert a flat array of global sequence values into a nested list grouped by sequence.

    Parameters
    ----------
    sequence_map : array-like
        Array containing the length of each sequence
    seqin360 : array-like
        Flat array containing all values from all sequences

    Returns
    -------
    list of lists
        Nested list where each inner list contains values for one sequence
    '''
    rsp = []
    offset = 0
    for seq in sequence_map:
        rsp.append(seqin360[offset:offset + int(seq)])
        offset = offset + int(seq)
    return rsp


def plot_confidences(ConfidencesPerSubject, title, Show=True, ExcludeUnanswered=True):
    '''
    Plot a histogram of confidence values.

    Parameters
    ----------
    ConfidencesPerSubject : array-like
        Confidence values to plot
    title : str
        Title for the plot
    Show : bool, optional
        Whether to display the plot. Default: True
    ExcludeUnanswered : bool, optional
        If True, exclude values of -1.0 (unanswered). Default: True

    Returns
    -------
    ndarray
        The processed confidence values
    '''
    ConfidencesPerSubject = numpy.asarray(ConfidencesPerSubject)
    confidences = ConfidencesPerSubject.flatten()

    # Unanswered responses are considered as zero for the voting mechanism.
    if ExcludeUnanswered:
        confidences = confidences[confidences != -1.0]
    else:
        confidences[confidences == -1.0] = 0.0

    val_confidences = numpy.arange(10.0 + 2.0, dtype=numpy.float32) / 10.0 - 0.05
    _conf_hist = numpy.histogram(confidences, bins=val_confidences.tolist())

    plt.hist(confidences, bins=val_confidences.tolist())
    plt.title(title)
    if Show:
        plt.show()

    return confidences
