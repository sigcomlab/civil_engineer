import scipy.io as sio
import numpy as np
import os

file_name = 'T_mat_0deg_192ch.mat'
file_path = os.path.join(os.pardir, 'calibration_matrices')
file_mat = os.path.join(file_path, file_name)

matrix = np.diag(sio.loadmat(file_mat)['T_mat_0deg'])
np.save(os.path.join(file_path, 'Sym_quad_0001.npy'), matrix)
print("Matrix saved")
