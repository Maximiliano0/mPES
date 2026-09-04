"""Sequence-splitting helper, mirrored from ``ml/pes_dqn/ext/tools.py``."""


def convert_globalseq_to_seqs(sequence_map, seqin360):
    """
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
    """
    rsp = []
    offset = 0
    for seq in sequence_map:
        rsp.append(seqin360[offset:offset + int(seq)])
        offset = offset + int(seq)
    return rsp
