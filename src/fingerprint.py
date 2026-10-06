"""Hex-encoded bit fingerprint decoding and Tanimoto similarity."""
import numpy as np


def hex_to_bits(hex_str: str) -> np.ndarray:
    """Decodes a hex-encoded bit fingerprint (e.g. Morgan/MACCS) to a 0/1 array."""
    raw = bytes.fromhex(hex_str)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits


def tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return float(intersection / union) if union > 0 else 0.0
