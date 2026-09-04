"""Self-contained evaluation harness for pes_ens_sprb.

Local copies of the Pandemic environment, severity dynamics, and result
formatting utilities, decoupled from ``ml.pes_dqn`` so this package never
imports TensorFlow-affecting model modules from ``ml/``.
"""
