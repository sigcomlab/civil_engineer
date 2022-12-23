from radar_manager.srs_manager import SRSManager
from radar_manager.sink import HdfSink
from radar_manager.velodyne_manager import LidarManager
from process.profiles_def import get_TI_ula_calib, Profiles
from qtplots import Generic3DScatterItem, Plot3D, Show2D, Generic2DLineItem
import numpy as np
from multiprocessing import set_start_method
from process.plotting import PolarProjection
from radar_manager.get_data import N_DOP

import time
import os
from utils.utils import root_dir
from datetime import datetime
now = datetime.now()
filename = now.strftime("%m-%d-%Y_%H-%M-%S_tornando_su.h5")


class Point:
    def __init__(self):
        self.range_idx = 0
        self.angle_idx = 0
        self.range = 0
        self.angle = 0


path_to_VA = str(os.path.join(root_dir, 'cfg_files', 'TI_VA_indices.npy'))
path_to_calib = str(os.path.join(root_dir, 'calibration_matrices', 'total_calibration_n512_fs12e6.npy'))

pair_VA_def = np.load(path_to_VA)
mat_calib = np.load(path_to_calib)


if __name__ == '__main__':
    set_start_method('spawn')

    with SRSManager('Srs_harvest_182', ip='', port=49000, description='central SRS radar') as SRS_C, \
            LidarManager('lidar_l', port=2369) as L, \
            HdfSink(filename, [SRS_C, L], autostart=True) as S:

        # This command is needed with SRS only
        while not SRS_C.is_ready:
            time.sleep(0.5)

        profile_c = Profiles(SRS_C, virtual_channels=np.arange(86), min_range=1,
                             max_range=40, range_fft_order=2048, cal_mat_file='none')

        SRS_C.display_params()

        plot3d = Plot3D({'apm': Generic3DScatterItem(profile_c.num_bin * profile_c.ang_fft_order, size=1),
                         'r': Generic3DScatterItem(1, size=10, color='r'),
                         'g': Generic3DScatterItem(1, size=10, color='g'),
                         'b': Generic3DScatterItem(1, size=10, color='b'),
                         'velo': Generic3DScatterItem(n_points=L.params['points'], color='g')
                         }, title='Radar scatter plot')




        aux_ap = np.zeros((profile_c.num_bin, profile_c.ang_fft_order, N_DOP), dtype='complex')

        tot_coords_c = profile_c.fft_coords
        plot3d['apm'].xyz = tot_coords_c.reshape(-1).cartesian_coords

        modifier = 'r'

        modifier_dict = dict()
        modifier_dict['r'] = Point()
        modifier_dict['g'] = Point()
        modifier_dict['b'] = Point()

        while plot3d.is_alive:

            framec = SRS_C.get_data()
            lidar_points = L.get_data().rotate(0, axis='z').translate([0.05, -2, 0])
            plot3d['velo'].xyz[:len(lidar_points)] = lidar_points.cartesian_coords[:len(lidar_points)]

            tpc = get_TI_ula_calib(framec, mat_calib, pair_VA_def)
            rpc = profile_c.range(tpc)
            ap = profile_c.angular(rpc, module=False)
            apc = np.log10(np.absolute(ap))


            tot_coords_c.intensity = apc[:, ::-1, -1].reshape(tot_coords_c.shape)

            plot3d['apm'].color = tot_coords_c.reshape(-1).intensity / tot_coords_c.intensity.max()


