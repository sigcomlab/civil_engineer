from radar_manager.srs_manager import SRSManager
from process.profiles_def import get_TI_ula_calib, Profiles
from process.plotting import PolarProjection
import time
import numpy as np
import os
from utils.utils import root_dir
from qtplots import Show2D, Generic2DLineItem, Generic3DScatterItem, Plot3D
from multiprocessing import set_start_method


if __name__ == '__main__':
    set_start_method('spawn')
    path_to_VA = str(os.path.join(root_dir, 'cfg_files', 'TI_VA_indices.npy'))
    path_to_calib = str(os.path.join(root_dir, 'calibration_matrices', 'total_calibration_n512_fs12e6.npy'))

    pair_VA_def = np.load(path_to_VA)
    mat_calib = np.load(path_to_calib)

    with SRSManager('srstest', ip='192.168.20.12', port=49000) as SRS:
        while not SRS.is_ready:
            time.sleep(0.5)
        SRS.display_params()

        time.sleep(1)
        profile_l = Profiles(SRS, virtual_channels=np.arange(86),
                             min_range=0.5,
                             max_range=6,
                             range_fft_order=2048,
                             ang_fft_order=128,
                             cal_mat_file='none')

        plot3d2 = Plot3D({'apm': Generic3DScatterItem(profile_l.num_bin * profile_l.ang_fft_order, size=1)},
                        title='Radar scatter plot')

        # Concatenate all the points in a single obj and set the ap plot coordinates accordingly to the radar
        tot_coords_c = profile_l.fft_coords.translate([0, -5.5, 0])

        plot3d2['apm'].xyz = tot_coords_c.reshape(-1).cartesian_coords

        # plot3d = PolarProjection(profile_l.range_scale, profile_l.angular_scale, title='CICCIO', db=False)

        plot2d = Show2D({'ph': Generic2DLineItem(1024, pen=(255, 255, 255, 255))},
                        title='phases')

        range_idx = 20
        ang_idx = 64
        t0 = time.perf_counter()
        ev = None
        while plot2d.is_alive:

            frame = SRS.get_data()
            tpl = get_TI_ula_calib(frame, mat_calib, pair_VA_def)
            rpl = profile_l.range(tpl)
            apl = profile_l.angular(rpl, module=False)

            # plot3d.update(np.absolute(apl[...]))

            apc = np.log10(profile_l.angular(rpl, module=True))

            tot_coords_c.intensity = apc[..., -1].reshape(tot_coords_c.shape)

            # plotting stuff
            plot3d2['apm'].color = tot_coords_c.reshape(-1).intensity / tot_coords_c.intensity.max()


            # ev = plot3d.get_event()

            if ev is not None:

                range_idx = profile_l.range_to_idx(ev[1])
                ang_idx = profile_l.angle_to_idx(ev[0], deg=False)
                print(range_idx, ang_idx)

            plot2d['ph'].append(np.angle(apl[range_idx, ang_idx]))
            time.sleep(0.01)


