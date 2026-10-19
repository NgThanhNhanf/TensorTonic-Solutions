import numpy as np
import math

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pass
    position = np.arange(0, seq_length, 1, dtype = np.float64)
    position = np.array([[p] for p in position], dtype = np.float64)
    pe = np.zeros((seq_length, d_model), dtype = np.float64)
    dev_term = np.exp(np.arange(0, d_model, 2, dtype = np.float64) * (-math.log(10000) / d_model))
    dev_term = np.array([dev_term])
    pe[:, 0::2] = np.sin(position * dev_term)
    pe[:, 1::2] = np.cos(position * dev_term)
    return pe