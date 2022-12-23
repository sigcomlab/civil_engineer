from radar_manager.x_manager import PhyObject
from utils.utils import precise_sleep, perf_counter
from coords import Coords3D
from numpy import frombuffer, array, pi, uint16, floor_divide, prod, copy, \
    empty, broadcast_to, ones, logical_not
from queue import Empty
import warnings
import socket
import time

TIMESTAMP_LEN = 8
COUNTER_LEN = 4


class LidarManager(PhyObject):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if exc_type:
            print('LidarManager:')
            print('exc_type: {}'.format(exc_type))
            print('exc_value: {}'.format(exc_val))
            print('exc_traceback: {}'.format(exc_tb))

    def __init__(self, str_id, ip='', port=2369, autostart=True,
                 description='default description', rate_reduction=0.05):
        super().__init__(str_id, dtype='B', mem_footprint=4096 * 196)
        self.ip = ip
        self.port = port
        self.description = description
        self._nloop = 151
        self._rate_reduction = rate_reduction

        # Mask to pick ranges from the packet raw bytes
        self._mask = ones(96, dtype=bool)
        self._mask[2::3] = False
        self._intensity_mask = logical_not(self._mask)

        self._asf = -18000 / pi  # azimuth scale factor
        esf = 180 / pi  # elevation scale factor

        lut = broadcast_to(array([-15, 1, -13, 3, -11, 5, -9, 7,
                                  -7, 9, -5, 11, -3, 13, -1, 15,
                                  -15, 1, -13, 3, -11, 5, -9, 7,
                                  -7, 9, -5, 11, -3, 13, -1, 15
                                  ]).reshape((1, 1, -1)) / esf, (self._nloop, 12, 16 * 2))

        self.pointcloud = Coords3D(shape=(12 * 32 * self._nloop))
        self.pointcloud.elevation[:] = lut.flatten()

        if autostart:
            self.start(self.run)

    def run(self):
        time_reference = time.time()    # take the time reference
        perf_cnt = perf_counter()

        params = dict()     # these will be the hdf files attributes to better understand the data
        params['type'] = 'velodyne lidar'
        params['description'] = self.description
        params['time_ref'] = time_reference
        params['perf_counter'] = perf_cnt
        params['laser_angles'] = array([-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15])
        params['points'] = self._nloop*12*32
        params['version'] = '0.3'
        params['shape'] = (self._nloop, 600)
        params['header_len'] = 12
        params['dtype'] = 'H'   # H is unsigned short int (16 bit)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((self.ip, self.port))

            # Wait for the first packet from lidar to initialize all internal parameters
            # and then make the socket non-blocking
            (payload, (lidar_ip, _)) = s.recvfrom(1206)
            data_blocks = copy(frombuffer(payload[:-6], dtype=uint16))
            temp_buffer = empty(((self._nloop,)+data_blocks.shape), dtype=uint16)
            s.setblocking(False)

            # initialize the shared memory region for Inter Process Communication
            # buffer_depth = int(floor_divide(self._shared_memory_size, prod(temp_buffer.shape)))
            # self.scb.init((buffer_depth,) + temp_buffer.shape)
            # print('{}: buffer initialized to shape {}'.format(self.str_id, (buffer_depth,) + temp_buffer.shape))

            mode = hex(payload[-2])
            model = hex(payload[-1])

            params['mode'] = mode
            params['model'] = model
            params['ip'] = lidar_ip

            elem_size = prod(params['shape']) * 2 + TIMESTAMP_LEN + COUNTER_LEN
            temp_elem = empty(elem_size, dtype='B')  # allocate the required memory once

            # compute the proper buffer size
            buffer_depth = int(floor_divide(self._shared_memory_size, elem_size))
            self.scb.init((buffer_depth,) + (elem_size,))
            print('{}: buffer initialized'.format(self.str_id))

            # create a reference for a float64 timestamp into the memory
            t0 = temp_elem[:TIMESTAMP_LEN]
            t0.dtype = 'd'  # double

            # create a reference for an int32 counter into the memory
            fc = temp_elem[TIMESTAMP_LEN:TIMESTAMP_LEN + COUNTER_LEN]
            fc.dtype = 'I'  # unsigned int

            # finally, reference the data
            data_flat = temp_elem[TIMESTAMP_LEN + COUNTER_LEN:]
            data_flat.dtype = 'H'   # unsigned short

            # Set the ready flag to true so everyone knows that the lidar object is ready
            self._is_ready.value = 1
            required = True

            while required:
                t_start = time.perf_counter()
                k = 0
                while k < self._nloop:
                    # receive packets from UDP
                    try:
                        payload = s.recv(65535)
                    except BlockingIOError:
                        # The socket is non-blocking to avoid freezing in case of loss of connection to the lidar.
                        # It raises BlockingIOError when no data is available
                        payload = b''
                    # if the message is empty skip part of the code
                    if payload == b'':
                        pass
                    else:
                        t0[0] = perf_counter()
                        fc[0] = self.counter.value
                        temp_buffer[k, ...] = frombuffer(payload[:-6], dtype='H')
                        # temp_buffer[k, 0] = uint16(((t0 - time_reference) * 1000) % (2 ** 15))
                        k += 1
                        if k == self._nloop:
                            data_flat[...] = temp_buffer.reshape(-1)

                            # send the new data to the circular buffer to make them available from the outside
                            # self.scb.write(temp_buffer.reshape((1,)+temp_buffer.shape))
                            self.scb.write(temp_elem.reshape((1, -1)))
                            self.is_last_read = False
                            # increment the frame counter
                            with self._external_lock:
                                self.counter.value += 1

                    # Always check if some command have been issued by the main process and handle them
                    try:
                        cmd = self._cmd_queue.get(block=False)[0]
                        if cmd == 'stop':
                            required = False
                        elif cmd == 'params':
                            self._ans_queue.put(params)
                        else:
                            warnings.warn('unknown command: ignored. Maybe you meant it for a player?', RuntimeWarning)
                    except Empty:
                        # slow down the cycle if necessary
                        while perf_counter() - t_start < self._rate_reduction:
                            precise_sleep(self._rate_reduction / 16)
