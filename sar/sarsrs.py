import time
from utils.ftdistep import FtdiStepper
from radar_manager.srs_manager import SRSManager
from process.profiles_def import get_TI_ula_calib, Profiles
from qtplots import Generic3DScatterItem, Plot3D, Show2D, Generic2DLineItem
import numpy as np
from multiprocessing import set_start_method
from process.plotting import PolarProjection
from radar_manager.get_data import N_DOP

import time
import os
from utils.utils import root_dir

ang_fft_order = 256
range_fft_order = 1024
d_ant = 0.5


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
    ste = FtdiStepper()
    ste.enable()
    with SRSManager('Srs_harvest_182', ip='', port=49000, description='central SRS radar') as SRS_C:

        # This command is needed with SRS only
        while not SRS_C.is_ready:
            time.sleep(0.5)

        profile_c = Profiles(SRS_C, virtual_channels=np.arange(86), min_range=0.5,
                             max_range=15, range_fft_order=2048, cal_mat_file='none', ang_fft_order=256)

        SRS_C.display_params()

        plot3d = Plot3D({'apm': Generic3DScatterItem(profile_c.num_bin * profile_c.ang_fft_order, size=1),
                         'r': Generic3DScatterItem(1, size=10, color='r'),
                         'g': Generic3DScatterItem(1, size=10, color='g'),
                         'b': Generic3DScatterItem(1, size=10, color='b')}, title='Radar scatter plot')

        full_angular_beamformed = Plot3D({'apm': Generic3DScatterItem(profile_c.num_bin * profile_c.ang_fft_order, size=1),
                         }, title='SAR scatter plot')




        plotpha3d = Plot3D(
            {'ang': Generic3DScatterItem(profile_c.num_bin * profile_c.ang_fft_order, cmap='cyclic', size=3),
             'r': Generic3DScatterItem(1, size=10, color='r'),
             'g': Generic3DScatterItem(1, size=10, color='g'),
             'b': Generic3DScatterItem(1, size=10, color='b')}, title='Phase scatter plot')

        plotphase = Show2D({'r': Generic2DLineItem(4096, pen=(255, 0, 0, 255)),
                            'g': Generic2DLineItem(4096, pen=(0, 255, 0, 255)),
                            'b': Generic2DLineItem(4096, pen=(0, 0, 255, 255))}, title='Phase plot vs time')

        plotfft = Show2D({'r': Generic2DLineItem(512, pen=(255, 0, 0, 255)),
                          'g': Generic2DLineItem(512, pen=(0, 255, 0, 255)),
                          'b': Generic2DLineItem(129, pen=(0, 0, 255, 255))}, title='FFT plot')

        # plotfft['r'].x = np.fft.rfftfreq(64*4, d=0.025)
        # plotfft['g'].x = np.fft.rfftfreq(64*4, d=0.025)
        # plotfft['b'].x = np.fft.rfftfreq(64 * 4, d=0.025)

        polarproj = PolarProjection(profile_c.range_scale, profile_c.angular_scale, title='PolarProj', db=False)
        phaseproj = PolarProjection(profile_c.range_scale, profile_c.angular_scale,
                                    title='PhaseBehavior', db=False, colorbar=[0, 3.14])

        aux_ap = np.zeros((profile_c.num_bin, profile_c.ang_fft_order, N_DOP), dtype='complex')

        tot_coords_c = profile_c.fft_coords
        plot3d['apm'].xyz = tot_coords_c.reshape(-1).cartesian_coords
        plotpha3d['ang'].xyz = tot_coords_c.reshape(-1).cartesian_coords
        full_angular_beamformed['apm'].xyz = tot_coords_c.reshape(-1).cartesian_coords

        modifier = 'r'

        modifier_dict = dict()
        modifier_dict['r'] = Point()
        modifier_dict['g'] = Point()
        modifier_dict['b'] = Point()

        frame_array = []
        framec = SRS_C.get_data()
        print(f'data shape:{framec.data.shape}')

        for step in range(142):
            ste.dosteps(113, period=0.0025)

            framec = SRS_C.get_data()

            tpc = get_TI_ula_calib(framec, mat_calib, pair_VA_def)

            frame_array.append(tpc)

            rpc = profile_c.range(tpc)
            ap = profile_c.angular(rpc, module=False)
            apc = np.log10(np.absolute(ap))
            polarproj.update(np.absolute(ap))

            phase_diff = np.angle(aux_ap * ap.conj())
            aux_ap[...] = ap
            ap_threshold = ap.max() / 5
            phase_diff[ap < ap_threshold] = 0

            phaseproj.update(np.absolute(phase_diff))

            tot_coords_c.intensity = apc[..., -1].reshape(tot_coords_c.shape)

            plot3d['apm'].color = tot_coords_c.reshape(-1).intensity / tot_coords_c.intensity.max()
            plotpha3d['ang'].color = (np.angle(ap[..., -1]).reshape(-1) + np.pi) / (2 * np.pi)
            ev = polarproj.get_event()
            if ev is not None:
                if ev[0] == 'm':
                    angle_idx = profile_c.angle_to_idx(ev[1].azimuth, deg=False)
                    range_idx = profile_c.range_to_idx(ev[1].range)
                    print(f'range {ev[1].range[0]}m, ang {ev[1].azimuth[0] / 3.14 * 180}deg. '
                          f'Indices = {[range_idx, angle_idx]}')

                    plot3d[modifier].x = ev[1].x
                    plot3d[modifier].y = ev[1].y

                    plotpha3d[modifier].x = ev[1].x
                    plotpha3d[modifier].y = ev[1].y

                    modifier_dict[modifier].angle_idx = angle_idx
                    modifier_dict[modifier].range_idx = range_idx

                    modifier_dict[modifier].angle = ev[1].azimuth
                    modifier_dict[modifier].range = ev[1].range

                    if modifier == 'r':
                        idx_modifier = 0
                    elif modifier == 'g':
                        idx_modifier = 1
                    elif modifier == 'b':
                        idx_modifier = 2

                    polarproj.addon_scatter[idx_modifier, :] = [ev[1].azimuth, ev[1].range]

                elif ev[0] == 'k':
                    if ev[1] in ['r', 'g', 'b']:
                        modifier = ev[1]
                        print('modifier modified to ', modifier)
                    elif ev[1] == 'o':
                        plotphase['r'].y -= plotphase['r'].y[-1]
                        plotphase['g'].y -= plotphase['g'].y[-1]
                        plotphase['b'].y -= plotphase['b'].y[-1]
                    else:
                        print('bad modifier, ignoring: ', ev[1])
            plotphase['r'].append(np.angle(ap[modifier_dict['r'].range_idx, modifier_dict['r'].angle_idx, 0]),
                                  unwrap=True)
            plotphase['g'].append(np.angle(ap[modifier_dict['g'].range_idx, modifier_dict['g'].angle_idx, 0]),
                                  unwrap=True)
            plotphase['b'].append(np.angle(ap[modifier_dict['b'].range_idx, modifier_dict['b'].angle_idx, 0]),
                                  unwrap=True)


            plotfft['r'].y = tpc[:, 0, 0].real
            plotfft['g'].y = tpc[:, 0, 0].imag


            # create a frame from the 86 channels:
            # frame_array_shape = [n_steps, Nsamples, 86, N_chirps]
            full_array = np.array(frame_array).sum(axis=2).transpose((1, 0, 2))
            full_array = full_array*np.broadcast_to(np.kaiser(full_array.shape[0], beta=6).reshape((-1,1,1)), full_array.shape)
            RP_sar = np.fft.fft(full_array, axis=0, n=profile_c.range_fft_order)[profile_c.idx_min_range:profile_c.idx_max_range, ...]
            RP_sar = RP_sar*np.broadcast_to(np.kaiser(full_array.shape[1], beta=6).reshape((1,-1,1)), RP_sar.shape)
            ap = np.fft.fftshift(np.fft.fft(RP_sar, axis=1, n=profile_c.ang_fft_order), axes=1)
            AP_sar = np.log10(np.absolute(ap))
            tot_coords_c.intensity = AP_sar[..., -1].reshape(tot_coords_c.shape)
            full_angular_beamformed['apm'].color = tot_coords_c.reshape(-1).intensity / tot_coords_c.intensity.max()
            print(step)
            time.sleep(0.5)




    ste.dosteps(-ste.STEPCOUNT, period=0.002)
    ste.disable()

    while plot3d.is_alive:
        time.sleep(0.5)
    np.save('magabigsarsoffitta2.npy', np.array(frame_array))
