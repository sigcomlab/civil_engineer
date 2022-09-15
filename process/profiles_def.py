from numpy import load, kaiser, fft, abs, arange, arcsin, pi, take, broadcast_to, matmul, linspace, \
                  sign, ones, absolute, meshgrid, exp, zeros,  mean, empty, newaxis
from radar_manager.get_data import N_SAMPLES, N_CHIPS, N_DOP, N_RX, N_TX
from numpy.linalg import eigh
from utils.utils import askopenfile, root_dir
from coords import Coords3D


import warnings
import time
import sys
import os


class Profiles:
    """
    Class that provides methods to compute range and angular profiles accordingly to the radar characteristics.
    """
    def __init__(self,
                 radar,                   # radar object
                 virtual_channels=arange(48),      # virtual channels of interest
                 min_range=0.5,
                 max_range=8,
                 cal_mat_file='auto',           # calibration matrix
                 range_window_order=3.8,          # kaiser coeff for range windowing
                 angular_window_order=4.2,        # kaiser coeff for ang windowing
                 doppler_window_order=4.5,
                 d_ant=0.5,                      # virtual antenna spacing [D/lambda]
                 ang_fft_order=256,
                 range_fft_order=4096,
                 elevation_fft_order=32,
                 doppler_fft_order=64,
                 music_order=128,
                 dump_first=64,
                 dump_last=32,      # discard some samples at the beginning and at the end of the chirp
                 angle_start=-pi / 2,
                 angle_end=pi / 2):
        """
        Initialize the Profile object
        :param radar: Radar object in use
        :param virtual_channels: virtual channels from the radar
        :param cal_mat_file: calibration matrix to search. 'auto' looks for a matrix called after the radar name
        :param range_window_order: range windowing order
        :param angular_window_order: angular windowing order
        :param d_ant: distance of the receiving antennas as fraction of lambda
        :param ang_fft_order: angular FFT order
        :param range_fft_order: range FFt order
        """

        self._radar = radar
        while not self._radar.is_ready:
            time.sleep(0.1)
        self.ts = 1 / self._radar.params['fs']
        self.kf = self._radar.params['bw'] / self._radar.params['trampup']      # [Hz/s]
        self.km = self.kf / 3e8     # [Hz/m]
        self.wavelength = 3e8 / self._radar.params['fcntr']

        self.dump_first = dump_first
        self.dump_last = dump_last

        self.virtual_channels = virtual_channels

        self.range_fft_order = range_fft_order
        self.range_fft_freq = arange(self.range_fft_order)/(self.range_fft_order*self.ts)  # fft.fftfreq(self.range_fft_order, self.ts)

        self.ang_fft_order = ang_fft_order
        self.angular_window_order = angular_window_order

        self.d_ant = d_ant
        self.ang_fft_freq = fft.fftshift(fft.fftfreq(self.ang_fft_order, self.d_ant))   # spatial frequency (normalized)

        self.elevation_fft_order = elevation_fft_order

        self.doppler_fft_order = doppler_fft_order
        self.doppler_window_order = doppler_window_order
        self.doppler_scale = fft.fftshift(
            fft.fftfreq(self.doppler_fft_order, self._radar.params['tframe']*self._radar.params['fcntr']/3e8))/2

        self.music_order = music_order
        # dummy values
        self.idx_min_range = 0
        self.idx_max_range = -1
        self.fft_coords = None

        self.range_scale = self.range_fft_freq * self.km * 2
        self.angular_scale = self.idx_to_angle(arange(self.ang_fft_order))

        self.angle_start = angle_start
        self.angle_end = angle_end

        self.speed_scale = self.idx_to_speed(arange(self.doppler_fft_order))

        # The following IF statement is required to pick the proper values from the calibration matrix in case of Symeo.
        if radar.params['frontend'] == 'ula0':  # or radar.params['frontend'] == 'ula':
            offset_virtual_idx = 48*2
        elif radar.params['frontend'] == 'ula1':
            offset_virtual_idx = 48*3
        elif radar.params['frontend'] == 'ula2':
            offset_virtual_idx = 48
        elif radar.params['frontend'] == 'ula3':
            offset_virtual_idx = 0
        else:
            offset_virtual_idx = 0

        # dirty but flexible way to find 'calibration_matrices' folder.
        cal_path = os.path.join(root_dir, 'calibration_matrices')
        self.calibration_path = cal_path
        try:
            if cal_mat_file == 'auto':
                cal_mat_file = radar.str_id
                self.cal_mat = take(load(os.path.join(cal_path, cal_mat_file + '.npy')),
                                    self.virtual_channels + offset_virtual_idx)
            elif cal_mat_file == 'none':
                self.cal_mat = ones(len(self.virtual_channels))
            elif cal_mat_file == 'pick':
                file = askopenfile(folder='calibration_matrices', extension='*.npy')
                if file == () or file == '':
                    warnings.warn('No calibration file specified: assuming none instead', RuntimeWarning)
                    self.cal_mat = ones(len(self.virtual_channels))
                else:
                    print('picking calibration file from: ', file)
                    self.cal_mat = take(load(file), self.virtual_channels + offset_virtual_idx)
            else:
                self.cal_mat = take(load(os.path.join(cal_path, cal_mat_file + '.npy')),
                                    self.virtual_channels + offset_virtual_idx)
        except FileNotFoundError:
            warnings.warn('Calibration file not found!', RuntimeWarning)
            sys.exit()

        dummy_data = self._radar.get_data()
        self.n_channels = len(self.virtual_channels)
        self.n_doppler = dummy_data.shape[-1]
        self.n_samples = dummy_data.shape[0] - dump_first - dump_last

        if self.n_samples > range_fft_order:
            self.n_samples = range_fft_order
        if self.n_channels > ang_fft_order:
            self.n_channels = ang_fft_order
        if self.n_doppler > doppler_fft_order:
            self.n_doppler = doppler_fft_order

        self.range_window_basic = kaiser(self.n_samples, beta=range_window_order).reshape((self.n_samples, 1, 1))
        self.cal_mat_basic = self.cal_mat.reshape((1, self.n_channels, 1))
        self.angular_window_basic = kaiser(self.n_channels, beta=angular_window_order).reshape((1, self.n_channels, 1))
        self.doppler_window_basic = kaiser(self.n_doppler, beta=doppler_window_order).reshape((1, 1, self.n_doppler))

        """
        MUSIC variables
        """
        self._antenna_range = arange(self.n_channels).reshape(-1, 1)
        self.music_angular_scale = linspace(self.angle_start, self.angle_end, music_order)
        self.music_angular_edges_scale = linspace(-pi/2-pi/music_order, pi/2+pi/music_order, music_order+1)
        self._a = exp(2j * self._antenna_range * self.music_angular_scale.reshape(1, -1))
        self.eig_matrix = None

        self.idx_max_range = self.update_max_range(max_range)
        print('idx_max_range', self.idx_max_range, max_range)
        self.idx_min_range = self.update_min_range(min_range)

    def range_to_idx(self, meters):
        """
        converts indices along the Range direction to meters
        :param meters: range
        :return: index corresponding to that range
        """
        # idx = (abs(self.range_fft_freq - meters * 2 * self.km)).argmin()
        idx = (abs(self.range_scale - meters)).argmin()
        return idx

    def idx_to_range(self, idx):
        """
        Converts index to range
        :param idx: index of interest
        :return: range in meters
        """
        meters = self.range_scale[idx.astype(int)]
        return meters

    def angle_to_idx(self, angle, deg=True):
        """
        convert angles to indices
        :param angle: angle. can be given in radiants or degrees. it must span between -90 to 90
        :param deg: True if angle is in degrees, False otherwise
        :return: index corresponding to specific angle
        """
        if deg:
            angle *= (pi / 180)
        idx = (abs(self.angular_scale - angle)).argmin()
        return idx

    def idx_to_angle(self, idx, deg=False, music=False):
        """
        Converts index to angle
        :param idx: index
        :param deg: True for angle in degrees, false for radiants
        :param music: True if your index refers to MUSIC
        :return: angle, in degree or radiants
        """
        if music:
            radiants = self.music_angular_scale[idx.astype(int)]
            if deg:
                return radiants*180/pi
            else:
                return radiants
        else:
            radiants = arcsin(self.ang_fft_freq[idx.astype(int)])
            if deg:
                return radiants * 180 / pi
            else:
                return radiants

    def speed_to_idx(self, speed):
        """
        convert speed to indices
        :param speed: speed in m/s
        :return: index corresponding to specific angle
        """

        idx = (abs(self.doppler_scale - speed)).argmin()
        return idx

    def idx_to_speed(self, idx):
        """
        Converts index to speed
        :param idx: index of interest
        :return: speed in m/s
        """
        speed = self.doppler_scale[idx.astype(int)]
        return speed

    def update_min_range(self, new_min_range):
        self.idx_min_range = (abs(self.range_fft_freq - new_min_range * 2 * self.km)).argmin()
        self._update_dependent_variables()
        return self.idx_min_range

    def update_max_range(self, new_max_range):
        self.idx_max_range = (abs(self.range_fft_freq - new_max_range * 2 * self.km)).argmin()
        self._update_dependent_variables()
        return self.idx_max_range

    def _update_dependent_variables(self):
        self.num_bin = self.idx_max_range - self.idx_min_range
        self.range_scale = self.range_fft_freq[self.idx_min_range:self.idx_max_range] / (2 * self.km)
        self.range_window = broadcast_to(self.range_window_basic, (self.n_samples, self.n_channels, self.n_doppler))
        self.angular_window = broadcast_to(self.angular_window_basic, (self.num_bin, self.n_channels, self.n_doppler))
        self.doppler_window = broadcast_to(self.doppler_window_basic,
                                           (self.num_bin, self.ang_fft_order, self.n_doppler))
        self.calibration_matrix = broadcast_to(self.cal_mat_basic, (self.num_bin, self.n_channels, self.n_doppler))
        self.fft_coords = Coords3D(shape=(self.num_bin, self.ang_fft_order))
        self.fft_coords.elevation = 0

        self.fft_coords.range = broadcast_to(self.range_scale.reshape(-1, 1), self.fft_coords.shape)
        self.fft_coords.azimuth = broadcast_to(self.angular_scale.reshape(1, -1), self.fft_coords.shape)

        self.coords = Coords3D(shape=(self.num_bin, self.ang_fft_order))

        self.coords.range = broadcast_to(self.range_scale.reshape(-1, 1), self.coords.shape)
        self.coords.azimuth = broadcast_to(self.angular_scale.reshape(1, -1), self.coords.shape)

        self.music_coords = Coords3D(shape=(self.num_bin, self.music_order))
        self.music_coords.elevation = 0

        # self.music_coords.range = broadcast_to(self.range_scale.reshape(-1, 1), self.music_coords.shape)
        # self.music_coords.azimuth = broadcast_to(self.music_angular_scale.reshape(1, -1), self.coords.shape)

    def chop(self, raw_data):
        """
        Discard samples at the beginning and at the end of the time sequence
        :param raw_data: raw data from the radar
        :return: raw_data purged along time axis
        """
        return raw_data[self.dump_first:-self.dump_last, self.virtual_channels, ...]

    def range(self, raw_data, calibrated=True):
        """
        Compute the range profile for given raw data in format [time_samples, doppler, virtual_channels]
        :param raw_data: raw data from ADC in the form
        :param calibrated: perform calibration using calibration matrix
        :return: Range profile in the form [range, angle, doppler (if available)]
        """
        if len(raw_data.shape) == 2:
            raw_data.shape = raw_data.shape+(1,)
        windowed_data = self.chop(raw_data-mean(raw_data))*self.range_window[..., :raw_data.shape[-1]]
        range_profile = fft.fft(windowed_data,
                                    n=self.range_fft_order,
                                    axis=0)[self.idx_min_range:self.idx_max_range, ...]
        # else:
        #     range_profile = fft.fft(windowed_data,
        #                             n=self.range_fft_order,
        #                             axis=0)[self.idx_min_range:self.idx_max_range, ...]
        if calibrated:

            range_profile *= self.calibration_matrix[..., :raw_data.shape[-1]]
        return range_profile

    def angular(self, range_profile, module=False, norm=False):
        """
        Compute the angular profile for a given Range Profile
        :param range_profile: input RP
        :param module:  compute the absolute value of the output
        :param norm: normalize the output between 0,1
        :return: Angular Profile
        """
        windowed_range_profile = range_profile * self.angular_window[..., :range_profile.shape[-1]]
        angular_profile = fft.fftshift(fft.fft(windowed_range_profile, n=self.ang_fft_order, axis=1), axes=1)

        if module and not norm:
            return absolute(angular_profile)  # [:, 1:, ...]
        elif norm:
            ap = absolute(angular_profile)  # [:, 1:, ...]
            return ap / ap.max()
        else:
            return angular_profile  # [:, 1:, ...]

    def ang_then_range(self, raw_data):
        """
        remember to remove doppler dimension beforehand
        """
        tp_chopped = self.chop(raw_data)
        tp_chopped_detrended = tp_chopped - tp_chopped.mean(axis=0)
        tp_calibrated = tp_chopped_detrended * broadcast_to(self.cal_mat_basic[..., 0], (self.n_samples, self.n_channels))
        tp_calibrated_windowed = tp_calibrated * broadcast_to(self.angular_window_basic[..., 0], (self.n_samples, self.n_channels))

        angular_fft = fft.fftshift(fft.fft(tp_calibrated_windowed, n=self.ang_fft_order, axis=1), axes=1)
        angular_fft_windowed = angular_fft*broadcast_to(self.range_window_basic[..., 0], (self.n_samples, self.ang_fft_order))
        ang_and_range_fft = fft.fft(angular_fft_windowed, n=self.range_fft_order, axis=0)[self.idx_min_range:self.idx_max_range, ...]
        return angular_fft, ang_and_range_fft

    def doppler(self, x_profile, from_range=True):
        """
        Compute the speed profile of the provided hypercube
        :param x_profile: input AP or RP, with non overlapping channels only
        :param from_range: true if the input is a Range Profile, false otherwise
        :return: doppler profile
        """
        if from_range:
            win_to_use = self.doppler_window[:, 0, :]
            x_profile = x_profile[:, 0, :]
            ax = 1
        else:
            win_to_use = self.doppler_window
            ax = 2
        windowed_angular_profile = x_profile*win_to_use
        doppler_profile = absolute(fft.fftshift(fft.fft(windowed_angular_profile, n=self.doppler_fft_order, axis=ax),
                                                axes=ax))
        return doppler_profile

    def music(self, range_profile, norm=False):
        """
        Compute the music pseudo-spectrum for a given Range Profile
        :param range_profile: input RP
        :param show_eigs: flag to show eigenvalues
        :param norm: normalize output wrt its maximum
        :return: MUSIC spectrum
        """
        # Eventually initialize eigenvalue plot
        # if show_eigs and self.eig_matrix is None:
        #     self.eig_matrix = MatrixShow(arange(self.n_channels+1), arange(self.idx_max_range-self.idx_min_range),
        #                                  title='Eigenvalues', xlabel='Eigenvalue', ylabel='Range bin', cm='cividis')

        # Discard the doppler dimension (if present)
        if len(range_profile.shape) >= 3:
            data = range_profile[..., -1]
        else:
            data = range_profile

        # Compute autocovariance as S=AA* when A is a column vector, and * is conjugate transpose operator
        s = data.reshape((-1, self.n_channels, 1)) * data.reshape((-1, 1, self.n_channels)).conj()

        # Compute eigenvalues and eigenvector in ascending order
        eigvals, eigvects = eigh(s)  # eigvect[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]

        # if show_eigs:
        #     self.eig_matrix.update(eigvals)

        # Take only the noise ones
        en = eigvects[..., :-1]

        # Apply MUSIC formula: P = 1/(a.T.conj() @ En @ En.T.conj() @ a).sum(axis=1)
        if not norm:
            return 1 / (matmul(
                matmul(broadcast_to(self._a.T.conj(), (data.shape[0], self.music_order, self.n_channels)),
                       matmul(en, en.transpose(0, 2, 1).conj())),
                broadcast_to(self._a,
                             (data.shape[0],
                              self.n_channels,
                              self.music_order)))).sum(axis=1)
        else:
            music_profile = absolute(1 / (matmul(
                matmul(broadcast_to(self._a.T.conj(), (data.shape[0], self.music_order, self.n_channels)),
                       matmul(en, en.transpose(0, 2, 1).conj())),
                broadcast_to(self._a,
                             (data.shape[0],
                              self.n_channels,
                              self.music_order)))).sum(axis=1))
            minimum = music_profile.min()
            maximum = music_profile.max()
            return (music_profile-minimum) / (maximum-minimum)

    def get_polygon_mask(self, vertices=Coords3D(), levels=None):
        """
        Generate a mask with 1 in the region delimited by the points, 0 elsewhere. The region must be non-convex.
        :param vertices: Vertices of the polygon
        :param levels: amplitude for outside and inside the mask
        :return: Matrix with the shape of AP[..., 0]
        """
        num_verts = len(vertices)
        if levels is None:
            levels = [0, 1]
        a_mask = ones((len(self.range_scale), len(self.angular_scale))) * levels[0]
        thetas, ranges = meshgrid(self.angular_scale, self.range_scale)
        all_the_coords = Coords3D(shape=len(thetas.reshape(-1)))
        all_the_coords.range = ranges.reshape(-1)
        all_the_coords.azimuth = thetas.reshape(-1)

        for i in range(num_verts):
            a_temp = zeros(len(thetas.reshape(-1)))   # temporary mask
            den = vertices.x[(i + 1) % num_verts] - vertices.x[i]   # denominator of rect equation
            if den == 0:
                if vertices.x[(i + 2) % num_verts] < vertices.x[i]:
                    a_temp[all_the_coords.x < vertices.x[i]] = levels[1]
                else:
                    a_temp[all_the_coords.x > vertices.x[i]] = levels[1]
            else:
                sign_den = sign(den)
                m = (vertices.y[(i + 1) % num_verts] - vertices.y[i]) / den
                q = vertices.y[i] - m * vertices.x[i]
                if sign_den < 0:
                    a_temp[(all_the_coords.y < m * all_the_coords.x + q)] = levels[1]
                else:
                    a_temp[(all_the_coords.y > m * all_the_coords.x + q)] = levels[1]
            a_mask *= a_temp.reshape(a_mask.shape)
        a_mask[a_mask == 0] = levels[0]
        return a_mask


def get_TI_ula(frame):

    pair_VA_def = load(str(os.path.join(root_dir, 'cfg_files', 'TI_VA_indices.npy')))

    frame_ula = empty((frame.data.shape[0], 86, frame.data.shape[-1]), dtype=complex)
    for jj in range(pair_VA_def.shape[1]):
        frame_ula[:, jj, :] = frame.data[..., pair_VA_def[0, jj], pair_VA_def[1, jj], pair_VA_def[2, jj], :]

    return frame_ula


def get_TI_ula_calib(frame, mat_calib, pair_VA_def):

    frame_calibed = frame.data*broadcast_to(mat_calib[..., newaxis], (N_SAMPLES, N_TX, N_RX, N_CHIPS, N_DOP))

    frame_ula = empty((frame.data.shape[0], 86, frame.data.shape[-1]), dtype=complex)
    for jj in range(pair_VA_def.shape[1]):
        frame_ula[:, jj, :] = frame_calibed[..., pair_VA_def[0, jj], pair_VA_def[1, jj], pair_VA_def[2, jj], :]

    return frame_ula
