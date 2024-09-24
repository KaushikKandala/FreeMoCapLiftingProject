import matplotlib.pyplot as plt
# import numpy as np
# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4])
# plt.show()
import numpy as np

# Load from .npy file
array = np.load(r"/Users/kaushik/Documents/EMSLifting/Data/EMSLifting/session_2024-07-25_15_33_57/recording_15_40_58_gmt-4/output_data/mediapipe_body_3d_xyz.npy")
print(array.shape)
print(array[:, 27, 2])
right_hip_z= array[:, 27, 2]
fig, ax = plt.subplots()
ax.plot(right_hip_z)
plt.show()
# Load from .npz file
