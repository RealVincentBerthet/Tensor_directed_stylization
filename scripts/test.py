import numpy as np



vect = np.array([1,2])

angle = np.rad2deg(np.arccos(np.clip(np.dot(np.array([0,1]), vect / np.linalg.norm(vect)), -1.0, 1.0)))

print(angle)