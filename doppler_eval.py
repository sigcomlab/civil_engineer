from radar_manager.player import Opener
from process.profiles_def import get_TI_ula_calib, Profiles
from qtplots import Generic3DScatterItem, Plot3D, Show2D, Generic2DLineItem
import numpy as np
from multiprocessing import set_start_method
from process.plotting import PolarProjection
import os
from utils.utils import root_dir, askopenfile
from datetime import datetime
import scipy.io as sio
from coords import Coords3D

now = datetime.now()
filename = now.strftime("%m-%d-%Y_%H-%M-%S_lab_civili.h5")


class Point(Coords3D):
    def __init__(self, range=0., angle=0., range_idx=0, angle_idx=0):
        super().__init__(shape=(1,))
        self.range_idx = range_idx
        self.angle_idx = angle_idx
        self.range = range
        self.azimuth = angle/180*np.pi


path_to_VA = str(os.path.join(root_dir, 'cfg_files', 'TI_VA_indices.npy'))
path_to_calib = str(os.path.join(root_dir, 'calibration_matrices', 'total_calibration_n512_fs12e6.npy'))

pair_VA_def = np.load(path_to_VA)
mat_calib = np.load(path_to_calib)

if __name__ == '__main__':
    set_start_method('spawn')
    filename = askopenfile('/run/media/giorgio')

    with Opener(filename, autorun=True, pause=True) as FILE:
        FILE.repr_speed = 0.1

        SRS_C = FILE['Srs_harvest_182']
        SRS_C.display_params()

        n_frames = SRS_C.n_frames
        phases_history = np.empty((n_frames, 4))
        amplitude_history = np.empty((n_frames, 4))
        doppler_history = np.empty((n_frames, 4))

        wavelength = 3e8 / SRS_C.params['fcntr']    # radar wavelength
        phase_to_speed = wavelength/(4*np.pi*SRS_C.params['tframe'])    # tframe is the time between two consecutive chirps

        # Profiles contains everything to compute range profiles, angular etc
        profile_c = Profiles(SRS_C, virtual_channels=np.arange(86), min_range=3.6,
                             max_range=6, range_fft_order=4096, cal_mat_file='none', dump_first=0, dump_last=0)

        # here is only to create the plots
        plot3d = Plot3D({'apm': Generic3DScatterItem(profile_c.num_bin * profile_c.ang_fft_order, size=1),
                         'r': Generic3DScatterItem(1, size=10, color='r'),
                         'g': Generic3DScatterItem(1, size=10, color='g'),
                         'b': Generic3DScatterItem(1, size=10, color='b')}, title='Radar scatter plot')

        plotphase = Show2D({'r': Generic2DLineItem(4096, pen=(255, 0, 0, 255)),
                            'g': Generic2DLineItem(4096, pen=(0, 255, 0, 255)),
                            'b': Generic2DLineItem(4096, pen=(0, 0, 255, 255))}, title='Phase plot vs time')

        plotampli = Show2D({'r': Generic2DLineItem(4096, pen=(255, 0, 0, 255)),
                            'g': Generic2DLineItem(4096, pen=(0, 255, 0, 255)),
                            'b': Generic2DLineItem(4096, pen=(0, 0, 255, 255))}, title='Amplitude plot vs time')

        plotdopp = Show2D({'r': Generic2DLineItem(4096, pen=(255, 0, 0, 255)),
                           'g': Generic2DLineItem(4096, pen=(0, 255, 0, 255)),
                           'b': Generic2DLineItem(4096, pen=(0, 0, 255, 255)),
                           }, title='doppler diffs')

        polarproj = PolarProjection(profile_c.range_scale, profile_c.angular_scale, title='PolarProj', db=False)

        tot_coords_c = profile_c.fft_coords     # radar complete pointcloud
        plot3d['apm'].xyz = tot_coords_c.reshape(-1).cartesian_coords

        modifier = 'r'

        modifier_dict = dict()
        modifier_dict['r'] = Point(3.709610142390809, -2.717782525851849, 16, 122)
        modifier_dict['g'] = Point(4.031413727229252, 12.167743038360735, 61, 155)
        modifier_dict['b'] = Point(4.533140304984563, -33.32319621134747, 131, 58)

        polarproj.addon_scatter[0, :] = [modifier_dict['r'].azimuth, modifier_dict['r'].range]
        polarproj.addon_scatter[1, :] = [modifier_dict['g'].azimuth, modifier_dict['g'].range]
        polarproj.addon_scatter[2, :] = [modifier_dict['b'].azimuth, modifier_dict['b'].range]

        plot3d['r'].x = modifier_dict['r'].x
        plot3d['r'].y = modifier_dict['r'].y

        plot3d['g'].x = modifier_dict['g'].x
        plot3d['g'].y = modifier_dict['g'].y

        plot3d['b'].x = modifier_dict['b'].x
        plot3d['b'].y = modifier_dict['b'].y

        k = 0
        while plot3d.is_alive and not FILE.eof:
            FILE.time += 0.01

            framec, ts = SRS_C.get_data(return_ts=True)     # return full frame and timestamp

            tpc = get_TI_ula_calib(framec, mat_calib, pair_VA_def)  # assemble the virtual array with 86 elements

            # do the range and angular FFT
            rp = profile_c.range(tpc, calibrated=False)
            ap = profile_c.angular(rp)
            # angular_fft, ap = profile_c.ang_then_range(tpc[..., 0])
            # angular_fft2, ap2 = profile_c.ang_then_range(tpc[..., 1])

            apc = np.log10(np.absolute(ap))
            polarproj.update(np.absolute(ap))

            tot_coords_c.intensity = apc[..., 0].reshape(tot_coords_c.shape)

            plot3d['apm'].color = tot_coords_c.reshape(-1).intensity / tot_coords_c.intensity.max()
            ev = polarproj.get_event()
            if ev is not None:
                if ev[0] == 'm':
                    angle_idx = profile_c.angle_to_idx(ev[1].azimuth, deg=False)
                    range_idx = profile_c.range_to_idx(ev[1].range)
                    print(f'{modifier} - range {ev[1].range[0]}m, ang {ev[1].azimuth[0] / 3.14 * 180}deg. '
                          f'Indices = {[range_idx, angle_idx]}')

                    plot3d[modifier].x = ev[1].x
                    plot3d[modifier].y = ev[1].y

                    modifier_dict[modifier].angle_idx = angle_idx
                    modifier_dict[modifier].range_idx = range_idx

                    modifier_dict[modifier].azimuth = ev[1].azimuth
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
                    elif ev[1] == 'm':
                        sio.savemat({'data': tpc})
                    else:
                        print('bad modifier, ignoring: ', ev[1])

            dopp_r = np.angle(ap[modifier_dict['r'].range_idx, modifier_dict['r'].angle_idx, 0] * ap[modifier_dict['r'].range_idx, modifier_dict['r'].angle_idx, 1].conj()) * phase_to_speed
            dopp_g = np.angle(ap[modifier_dict['g'].range_idx, modifier_dict['g'].angle_idx, 0] * ap[modifier_dict['g'].range_idx, modifier_dict['g'].angle_idx, 1].conj()) * phase_to_speed
            dopp_b = np.angle(ap[modifier_dict['b'].range_idx, modifier_dict['b'].angle_idx, 0] * ap[modifier_dict['b'].range_idx, modifier_dict['b'].angle_idx, 1].conj()) * phase_to_speed

            plotdopp['r'].append(dopp_r, FILE.time)
            plotdopp['g'].append(dopp_g, FILE.time)
            plotdopp['b'].append(dopp_b, FILE.time)

            phase_r = np.angle(ap[modifier_dict['r'].range_idx, modifier_dict['r'].angle_idx, 0])
            phase_g = np.angle(ap[modifier_dict['g'].range_idx, modifier_dict['g'].angle_idx, 0])
            phase_b = np.angle(ap[modifier_dict['b'].range_idx, modifier_dict['b'].angle_idx, 0])

            plotphase['r'].append(phase_r, FILE.time, unwrap=True)
            plotphase['g'].append(phase_g, FILE.time, unwrap=True)
            plotphase['b'].append(phase_b, FILE.time, unwrap=True)

            ampli_r = np.absolute(ap[modifier_dict['r'].range_idx, modifier_dict['r'].angle_idx, 0])
            ampli_g = np.absolute(ap[modifier_dict['g'].range_idx, modifier_dict['g'].angle_idx, 0])
            ampli_b = np.absolute(ap[modifier_dict['b'].range_idx, modifier_dict['b'].angle_idx, 0])

            plotampli['r'].append(ampli_r, FILE.time)
            plotampli['g'].append(ampli_g, FILE.time)
            plotampli['b'].append(ampli_b, FILE.time)

            doppler_history[k, :] = FILE.time, dopp_r, dopp_g, dopp_b
            phases_history[k, :] = FILE.time, phase_r, phase_g, phase_b
            amplitude_history[k, :] = FILE.time, ampli_r, ampli_g, ampli_b



            k += 1

        sio.savemat(filename[:-4]+'.mat', {'doppler_history': doppler_history,
                                           'phases_history': phases_history,
                                           'amplitude_history': amplitude_history})






