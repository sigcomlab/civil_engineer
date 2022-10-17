from radar_manager.player import Player, Opener
from process.profiles_def import get_TI_ula_calib, Profiles
from qtplots import Generic3DScatterItem, Plot3D, Show2D, Generic2DLineItem
import numpy as np
from multiprocessing import set_start_method
from process.plotting import PolarProjection
from radar_manager.get_data import N_DOP

from process.CSFEC import CSFEC, const
import os
from utils.utils import root_dir, askopenfile
from datetime import datetime
now = datetime.now()
filename = now.strftime("%m-%d-%Y_%H-%M-%S_lab_civili.h5")
import scipy.io as sio


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
    filename = askopenfile('/run/media/giorgio')

    with Opener(filename, autorun=True, pause=True) as FILE:
        FILE.repr_speed = 0.3
        SRS_C = FILE['Srs_harvest_182']
        FILE.pause = False

        # This command is needed with SRS only
        profile_c = Profiles(SRS_C, virtual_channels=np.arange(86), min_range=2.5,
                             max_range=6.5, range_fft_order=4096, cal_mat_file='none', dump_first=0, dump_last=0)

        plot3d = Plot3D({'apm': Generic3DScatterItem(profile_c.num_bin * profile_c.ang_fft_order, size=1),
                         'r': Generic3DScatterItem(1, size=10, color='r'),
                         'g': Generic3DScatterItem(1, size=10, color='g'),
                         'b': Generic3DScatterItem(1, size=10, color='b')}, title='Radar scatter plot')

        plotpha3d = Plot3D({'ang': Generic3DScatterItem(profile_c.num_bin * profile_c.ang_fft_order, cmap='cyclic', size=3),
                            'r': Generic3DScatterItem(1, size=10, color='r'),
                            'g': Generic3DScatterItem(1, size=10, color='g'),
                            'b': Generic3DScatterItem(1, size=10, color='b')}, title='Phase scatter plot')

        plotphase = Show2D({'r': Generic2DLineItem(4096, pen=(255, 0, 0, 255)),
                            'g': Generic2DLineItem(4096, pen=(0, 255, 0, 255)),
                            'b': Generic2DLineItem(4096, pen=(0, 0, 255, 255))}, title='Phase plot vs time')

        plotampli = Show2D({'r': Generic2DLineItem(4096, pen=(255, 0, 0, 255)),
                            'g': Generic2DLineItem(4096, pen=(0, 255, 0, 255)),
                            'b': Generic2DLineItem(4096, pen=(0, 0, 255, 255))}, title='Amplitude plot vs time')

        plotcsfe = Show2D({'r0': Generic2DLineItem(4096, pen=(255, 0, 0, 255)),
                           'g0': Generic2DLineItem(4096, pen=(0, 255, 0, 255)),
                           'b0': Generic2DLineItem(4096, pen=(0, 0, 255, 255)),

                           'r1': Generic2DLineItem(4096, pen=(128, 0, 0, 255)),
                           'g1': Generic2DLineItem(4096, pen=(0, 128, 0, 255)),
                           'b1': Generic2DLineItem(4096, pen=(0, 0, 128, 255)),

                           'r2': Generic2DLineItem(4096, pen=(64, 0, 0, 255)),
                           'g2': Generic2DLineItem(4096, pen=(0, 64, 0, 255)),
                           'b2': Generic2DLineItem(4096, pen=(0, 0, 64, 255))
                           }, title='csfe freqs')

        polarproj = PolarProjection(profile_c.range_scale, profile_c.angular_scale, title='PolarProj', db=False)
        phaseproj = PolarProjection(profile_c.range_scale, profile_c.angular_scale,
                                    title='PhaseBehavior', db=False, colorbar=[0, 3.14])

        aux_ap = np.zeros((profile_c.num_bin, profile_c.ang_fft_order), dtype='complex')

        tot_coords_c = profile_c.fft_coords
        plot3d['apm'].xyz = tot_coords_c.reshape(-1).cartesian_coords
        plotpha3d['ang'].xyz = tot_coords_c.reshape(-1).cartesian_coords

        modifier = 'r'

        modifier_dict = dict()
        modifier_dict['r'] = Point()
        modifier_dict['g'] = Point()
        modifier_dict['b'] = Point()

        # freq = idx/N0 * fs --->  idx = freq/fs*N0
        M = 8
        min_distance = 3.8  # meters
        max_distance = 4.2  # meters

        # csfe_angles = {'r': -5., 'g': 0., 'b': 5.}

        min_freq = min_distance * 2 * profile_c.km
        min_csfe_idx = int(min_freq * profile_c.ts * 512 * M)
        max_freq = max_distance * 2 * profile_c.km
        max_csfe_idx = int(max_freq * profile_c.ts * 512 * M)
        csfe_constants = [const(K=3, N=512, M=8, minmaxidx=[min_csfe_idx, max_csfe_idx])]*3
        csfe_angles = [-5., 0., 5.]

        while plot3d.is_alive:

            framec = SRS_C.get_data()

            tpc = get_TI_ula_calib(framec, mat_calib, pair_VA_def)[..., -1]

            angular_fft, ap = profile_c.ang_then_range(tpc)

            apc = np.log10(np.absolute(ap))
            polarproj.update(np.absolute(ap))

            phase_diff = np.angle(aux_ap*ap.conj())
            aux_ap[...] = ap
            ap_threshold = ap.max()/5
            phase_diff[ap < ap_threshold] = 0

            phaseproj.update(np.absolute(phase_diff))

            tot_coords_c.intensity = apc[...].reshape(tot_coords_c.shape)

            plot3d['apm'].color = tot_coords_c.reshape(-1).intensity / tot_coords_c.intensity.max()
            plotpha3d['ang'].color = (np.angle(ap[...]).reshape(-1) + np.pi)/(2*np.pi)
            ev = polarproj.get_event()
            if ev is not None:
                if ev[0] == 'm':
                    angle_idx = profile_c.angle_to_idx(ev[1].azimuth, deg=False)
                    range_idx = profile_c.range_to_idx(ev[1].range)
                    print(f'range {ev[1].range[0]}m, ang {ev[1].azimuth[0]/3.14*180}deg. '
                          f'Indices = {[range_idx, angle_idx]}')

                    min_distance = ev[1].range[0] - 0.2
                    max_distance = ev[1].range[0] + 0.2

                    min_freq = min_distance * 2 * profile_c.km
                    min_csfe_idx = int(min_freq * profile_c.ts * 512 * M)
                    max_freq = max_distance * 2 * profile_c.km
                    max_csfe_idx = int(max_freq * profile_c.ts * 512 * M)

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

                    # change the angle on which perform the csfe, and also its relative const.
                    csfe_angles[idx_modifier] = ev[1].azimuth[0]/3.14*180
                    csfe_constants[idx_modifier] = const(K=3, N=tpc.shape[0], M=8, minmaxidx=[min_csfe_idx, max_csfe_idx])

                    polarproj.addon_scatter[idx_modifier, :] = [ev[1].azimuth, ev[1].range]
                    phaseproj.addon_scatter[idx_modifier, :] = [ev[1].azimuth, ev[1].range]

                elif ev[0] == 'k':
                    if ev[1] in ['r', 'g', 'b']:
                        modifier = ev[1]
                        print('modifier modified to ', modifier)
                    elif ev[1] == 'o':
                        plotphase['r'].y -= plotphase['r'].y[-1]
                        plotphase['g'].y -= plotphase['g'].y[-1]
                        plotphase['b'].y -= plotphase['b'].y[-1]
                    elif ev[1] == 'm':
                        sio.savemat()
                    else:
                        print('bad modifier, ignoring: ', ev[1])
            # do csfe
            for angle, csfe_const, mod in zip(csfe_angles, csfe_constants, ['r', 'g', 'b']):
                csfe_angle_idx = profile_c.angle_to_idx(angle, deg=True)
                ampls, freqs = CSFEC(angular_fft[:, int(csfe_angle_idx)], csfe_const)
                plotcsfe[mod + '0'].append(freqs[0] / (2 * profile_c.km * profile_c.ts))
                plotcsfe[mod + '1'].append(freqs[1] / (2 * profile_c.km * profile_c.ts))
                plotcsfe[mod + '2'].append(freqs[2] / (2 * profile_c.km * profile_c.ts))

            plotphase['r'].append(np.angle(ap[modifier_dict['r'].range_idx, modifier_dict['r'].angle_idx]), unwrap=True)
            plotphase['g'].append(np.angle(ap[modifier_dict['g'].range_idx, modifier_dict['g'].angle_idx]), unwrap=True)
            plotphase['b'].append(np.angle(ap[modifier_dict['b'].range_idx, modifier_dict['b'].angle_idx]), unwrap=True)

            plotampli['r'].append(np.absolute(ap[modifier_dict['r'].range_idx, modifier_dict['r'].angle_idx]))
            plotampli['g'].append(np.absolute(ap[modifier_dict['g'].range_idx, modifier_dict['g'].angle_idx]))
            plotampli['b'].append(np.absolute(ap[modifier_dict['b'].range_idx, modifier_dict['b'].angle_idx]))








