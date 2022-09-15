import numpy as np
from coords import Coords3D
from cv2 import imdecode, IMREAD_COLOR

N_RX = 4
N_TX = 12
N_DOP = 2
N_CHIPS = 4
N_SAMPLES = 512

R = 0
I = 1

TIMESTAMP_LEN = 8
COUNTER_LEN = 4

packet_size = 8210  # 6+bytes_to_read_after_header*(512//N_SAMPLES)

packets_per_frame = N_DOP*N_TX*N_CHIPS


class SubFrame:
    def __init__(self, bytebuffer):
        npbuf = bytebuffer
        npbuf.dtype = '>H'
        self.iq = npbuf[(-6 - N_SAMPLES * 8):-6]
        self.iq.dtype = 'h'  # counters etc. are big endian, data are little endian
        self.tail = npbuf[-6:-4]
        self._sequence_number = npbuf[-4:-3]    # this is an array with one element
        self.sequence_number = npbuf[-4]         # this is an int()
        self.chirp_number = npbuf[-3]-1
        self.chip_number = npbuf[-2]
        self.checksum = npbuf[-1]
        #
        # # the followings are a nice trick to assert which frame a packet(i.e. a subframe) belongs to.
        self._frame_number = self._sequence_number - (self.chirp_number*4 + self.chip_number)    # result is an int16 array with one element
        #
        self._frame_number //= 4  # self.frame_number>>2
        if self._frame_number[0] % 2 != 0:
            self._frame_number += 1
        self.frame_number = self._frame_number[0]//12  # this is int() again

    def __repr__(self):
        return 'SubFrame {} {}, seq num: {}, chirp_num: {}, chip_num: {}, frame_number: {}'.format(hex(self.tail[0]),
                                                                                                  hex(self.tail[1]),
                                                                                                  self.sequence_number,
                                                                                                  self.chirp_number,
                                                                                                  self.chip_number,
                                                                                                  self.frame_number)


class Frame:
    def __init__(self, subframe=None):
        self.timestamp = 0    # Use the moment at which this Frame is created as timestamp.
        self.frame_number = 0
        self.checksum = 0

        # keep real and imag part a int16 will save 2 to 4 times disk space.
        self.data = np.empty((N_SAMPLES, N_TX, N_RX, N_CHIPS, N_DOP), dtype='complex')
        self._subframes = dict()
        self.obsolescence = 0
        self._n_written = 0
        self.num_subframes_per_frame = N_DOP * N_TX * N_CHIPS

        if subframe is not None:
            self.fill_from_subframe(subframe)

    @property
    def shape(self):
        return self.data.shape

    def is_complete(self):
        return self._n_written == self.num_subframes_per_frame

    def fill_from_subframe(self, subframe):
        if self.frame_number == 0:
            self.frame_number = subframe.frame_number
        if self.frame_number != subframe.frame_number:
            pass
            # print('Sequence number mismatch', self.frame_number, subframe.frame_number)
            # raise ValueError('Sequence number mismatch')

        """
        Here we are assuming the following transmitting scheme: 
        DOP0: Tx0 Tx1 Tx2 Tx3...
        DOP1: Tx0 Tx1 .....
        DOP2: ......
        """
        # n_dop = subframe.chirp_number % N_DOP  # Figure out what TX antenna was active in this subframe...
        # n_tx = subframe.chirp_number // N_DOP  # ...and what doppler frame are we dealing with.
        n_tx = subframe.chirp_number % N_TX  # Figure out what TX antenna was active in this subframe...
        n_dop = subframe.chirp_number // N_TX  # ...and what doppler frame are we dealing with.
        chip = subframe.chip_number

        self.data[:, n_tx, 0, chip, n_dop] = subframe.iq[::8]+1j*subframe.iq[1::8]
        self.data[:, n_tx, 1, chip, n_dop] = subframe.iq[2::8]+1j*subframe.iq[3::8]
        self.data[:, n_tx, 2, chip, n_dop] = subframe.iq[4::8]+1j*subframe.iq[5::8]
        self.data[:, n_tx, 3, chip, n_dop] = subframe.iq[6::8]+1j*subframe.iq[7::8]

        return self.num_subframes_per_frame-self._n_written


def get_data_by_type(self, row, return_ts):
    if 'header_len' in self.params.keys():
        header_len = self.params['header_len']
    else:
        header_len = 12     # fallback to maintain retrocompatibility
    data = row[header_len:]
    data.dtype = self.params['dtype']
    data.shape = self.params['shape']

    if self.params['type'] == 'velodyne lidar':

        nloop = self.params['shape'][0]
        asf = -18000 / np.pi   # amplitude scale factor
        esf = 180 / np.pi  # elevation scale factor
        lut = np.broadcast_to(np.asarray([self.params['laser_angles'],
                                          self.params['laser_angles']]).reshape((1, 1, -1)) / esf, (nloop, 12, 16 * 2))
        pointcloud = Coords3D(shape=self.params['points'])
        pointcloud.elevation[:] = lut.reshape(-1)

        mask = np.ones(96, dtype=bool)
        mask[2::3] = False
        intensity_mask = np.logical_not(mask)

        for k in range(nloop):
            # Insert into a buffer: 12 "Data Blocks" 100 bytes each
            data_blocks = data[k, ...].view(dtype='B').reshape(12, 100, order='C')

            # Create the azimuth table
            w = pointcloud.azimuth[k * 12 * 32:(k + 1) * 12 * 32].view().reshape(12, 32)
            w[:, :16] = np.broadcast_to(np.copy(data_blocks[:, 2:4]).view('H') / asf, (12, 16))
            w[:, 16:] = np.gradient(w[:, :16], axis=1) / 2 + w[:, :16]

            # Some magics to the bytes
            pointcloud.range[k * 12 * 32:(k + 1) * 12 * 32] = \
                np.copy((data_blocks[:, 4:][..., mask]).T, order='F').T.reshape(-1).view('H') / 500

            # Compute the 3D pointcloud
            pointcloud.intensity[k * 12 * 32:(k + 1) * 12 * 32] = \
                np.copy((data_blocks[:, 4:][..., intensity_mask])).reshape(-1)
        to_return = pointcloud

    elif self.params['type'] == 'septentrio gps':
        # if this is the GPS, return GGA and HDT strings only.
        strings = ''.join([chr(c) for c in data]).split('\r\n')
        to_return = strings[:min(2, len(strings))]

    elif self.params['type'] == 'webcam':
        to_return = imdecode(data, IMREAD_COLOR)[..., ::-1]

    elif self.params['type'] == 'ouster lidar':
        to_return = Coords3D(data.shape[0])
        to_return.cartesian_coords = data[:, :3]/1000
        to_return.intensity = data[:, 3]

    elif self.params['type'] == 'SRS radar':
        valz = data
        frame = Frame()
        for idx in range(packets_per_frame):
            subframe0 = SubFrame(valz[packet_size * idx:packet_size * (idx + 1)])
            frame.fill_from_subframe(subframe0)
        ts = row[:8]
        ts.dtype = 'd'
        frame.timestamp = ts
        to_return = frame

    else:
        to_return = data

    if return_ts:
        ts = row[:8]
        ts.dtype = 'd'
        return to_return, return_ts
    else:
        return to_return
