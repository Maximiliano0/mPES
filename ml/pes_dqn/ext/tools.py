'''
pes_dqn - Pandemic Experiment Scenario: Utility Functions

Small collection of helper functions used across pes_dqn modules:

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
