# Convert coordinates from MNI305 space to MNI152 space

import numpy as np

# Transformation matrix
# Ref: https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
M = np.array([
    [0.9975,   -0.0073,    0.0176,   -0.0429],
    [0.0146,    1.0009,   -0.0024,    1.5496],
    [-0.0130,   -0.0093,    0.9971,    1.1840]
    ])

# Specify your RAS coordinates in MNI305 space
v = np.array([29, -40, -15, 1])
v = v.transpose()

# Tranform to MNI152 space
print(np.matmul(M, v))

# you can then input these coordinates into image viewing software (e.g. MRICron, xjview)
# to read out the corresponding anatomical label
