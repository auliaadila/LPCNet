import numpy as np

N = 160  # samples per frame
M = 64  # bits per frame
rng = np.random.default_rng(seed=2025)

# Step 1: start from an NxN Hadamard (power of two) and keep first 64 rows
H = np.sign(
    np.array(
        np.fromfunction(
            lambda i, j: (-1) ** bin(i & j).count("1"), (128, 128), dtype=int
        )
    )
)  # 128×128 Walsh–Hadamard
codes = np.zeros((M, N))
print(codes.shape)
codes[:, :128] = H[:M]  # pad last 32 samples with periodic wrap
print(codes.shape)
codes[:, 128:] = H[:M, :32]
print(codes.shape)

# Normalise energy: each code has variance 1/N
codes /= np.sqrt(N)
print(codes.shape)

# import scipy.linalg as la
# H = la.hadamard(128)        # entries are ±1
# codes64 = H[:64]            # first 64 rows
# print(H.shape)
# print(codes64.shape)
