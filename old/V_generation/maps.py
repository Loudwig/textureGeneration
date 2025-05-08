import numpy as np

import matplotlib.pyplot as plt

# Create a 3x3 matrix
matrix = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
# Create three more matrices
matrix2 = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
matrix3 = np.array([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]])
matrix4 = np.array(np.transpose([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]))

# Create a figure with subplots
fig, axs = plt.subplots(2, 2)

# Plot each matrix in a subplot
axs[0, 0].imshow(matrix, cmap="Greys")
axs[0, 1].imshow(matrix2, cmap="Greys")
axs[1, 0].imshow(matrix3, cmap="Greys")
axs[1, 1].imshow(matrix4, cmap="Greys")
# Add colorbars to each subplot
axs[0, 0].set_title("Matrix 1")
axs[0, 1].set_title("Matrix 2")
axs[1, 0].set_title("Matrix 3")
axs[1, 1].set_title("Matrix 4")

# Show the plott
plt.show()
# Change the colormap
