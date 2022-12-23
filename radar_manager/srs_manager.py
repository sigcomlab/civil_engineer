import numpy as np
from radar_manager.x_manager import PhyObject
import socket
from multiprocessing import Value
from time import perf_counter
import time
import logging
from queue import Empty

from radar_manager.get_data import N_CHIPS, N_DOP, N_TX, packet_size

TIMESTAMP_LEN = 8
COUNTER_LEN = 4

packets_per_frame = N_DOP*N_TX*N_CHIPS


class SRSManager(PhyObject):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if exc_type:
            print('SRSManager:')
            print('exc_type: {}'.format(exc_type))
            print('exc_value: {}'.format(exc_val))
            print('exc_traceback: {}'.format(exc_tb))

    def __init__(self, str_id, ip='', port=49000, description='default description', autostart=True):
        super().__init__(str_id, dtype='B', mem_footprint=256*1024*1024)
        self.ip = ip
        self.port = port
        self.description = description
        self._required = Value('i')
        self._required.value = 1
        if autostart:
            self.start()

    def _run(self):
        time_reference = time.time()  # get the time reference for the data
        perf_cnt = perf_counter()
        params = dict()
        params['type'] = 'SRS radar'
        params['fs'] = 6250e3
        params['fcntr'] = 77.5e9
        params['trampup'] = 92e-6
        params['tframe'] = 100e-6  # (idle time+ ramp end time)
        params['shape'] = (788160,)  # (N_SAMPLES, N_TX, N_RX, N_CHIPS, N_DOP, 2)
        params['dtype'] = 'B'
        params['bw'] = 32*92*1e6
        params['frontend'] = 'ti'
        params['description'] = self.description
        params['time_ref'] = time_reference
        params['perf_counter'] = perf_cnt
        params['header_len'] = 12
        params['version'] = '0.3'
        params['packets_per_frame'] = packets_per_frame
        params['packet_size'] = packet_size

        full_frame_size_bytes = packet_size * packets_per_frame
        elem_size = packet_size * packets_per_frame + TIMESTAMP_LEN + COUNTER_LEN
        temp_elem = np.empty((1, elem_size), dtype='B')  # allocate the required memory once
        t0 = temp_elem[0, :TIMESTAMP_LEN]
        t0.dtype = 'd'  # double

        # create a reference for an int32 counter into the memory
        fc = temp_elem[0, TIMESTAMP_LEN:TIMESTAMP_LEN + COUNTER_LEN]
        fc.dtype = 'I'  # unsigned int

        # fcr = temp_elem[TIMESTAMP_LEN + COUNTER_LEN:TIMESTAMP_LEN + COUNTER_LEN + RADAR_COUNTER_LEN]
        # fcr.dtype = 'I'  # 32 bit unsigned int

        # finally, reference the data
        data_flat = temp_elem[0, TIMESTAMP_LEN + COUNTER_LEN:]
        data_flat.dtype = 'B'  # unsigned short
        print('srs ready')

        # self.scb = CircularBuffer(2**28, dtype='B')
        buffer_depth = int(np.floor_divide(self._shared_memory_size, elem_size))
        self.scb.init((buffer_depth,) + (elem_size,))
        print('scb shape', self.scb.shape, self.scb.elem_shape)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((self.ip, self.port))
            s.settimeout(0.02)
            self._is_ready.value = 1

            # Allocate a buffer for 48 packets
            buf = bytearray(packet_size * packets_per_frame)
            full_frame = False  # aux variable

            # Store older packets due to defect in SRS code
            stupid_buffer = np.empty(packet_size * (packets_per_frame - 1), dtype='B')
            is_stupid_buffer_initialized = False
            while self._required.value:
                view = memoryview(buf)
                toread = packet_size * packets_per_frame
                try:
                    while toread:
                        nbytes = s.recv_into(view, packet_size)
                        view = view[nbytes:]  # slicing views is cheap
                        toread -= nbytes

                    full_frame = True

                except socket.timeout:
                    if not is_stupid_buffer_initialized:
                        stupid_buffer = np.frombuffer(buf[-packet_size * (packets_per_frame - 1):], dtype='B')
                        is_stupid_buffer_initialized = True
                    if full_frame:
                        t0[0] = perf_counter()
                        fc[0] = self.counter.value
                        self.counter.value += 1
                        # print('full frame received')

                        data_flat[..., packet_size * (packets_per_frame - 1):] = np.frombuffer(
                            buf[:-packet_size * (packets_per_frame - 1)], dtype='B')
                        data_flat[..., :packet_size * (packets_per_frame - 1)] = stupid_buffer
                        stupid_buffer = np.frombuffer(buf[-packet_size * (packets_per_frame - 1):], dtype='B')

                        self.scb.write(temp_elem)
                        self._is_last_read = False
                        full_frame = False
                    else:
                        if toread != full_frame_size_bytes:
                            pass
                            print('packet loss', toread, full_frame_size_bytes)

                    try:
                        cmd = self._cmd_queue.get(block=False)[0]
                        if cmd == 'stop':
                            self._required.value = False
                        elif cmd == 'params':
                            self._ans_queue.put(params)
                        else:
                            logging.warning('SRS {}: Unknown command: ignored. '.format(self.str_id))
                    except Empty:
                        pass
