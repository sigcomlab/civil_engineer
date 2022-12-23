import h5py
import numpy as np
import multiprocessing as mp
from utils.utils import precise_sleep
from radar_manager.get_data import get_data_by_type
import time


def canparser(filename, group, objs):
    with h5py.File(filename, 'r+') as F:
        print('File opened: \n{}'.format(filename))
        src_grp = F[group]
        src_dset = src_grp['raw_data']
        next_step_objs = []
        for obj in objs:
            try:
                dst_grp = F.create_group(obj.type)
                print('Created {} group'.format(obj.type))
                dst_dset = dst_grp.create_dataset(name='raw_data',
                                                  shape=(0,) + src_dset.shape[1:],
                                                  dtype='B',
                                                  maxshape=(None,) + src_dset.shape[1:])
                next_step_objs.append(obj)
                for attr in src_dset.attrs.keys():
                    dst_dset.attrs[attr] = src_dset.attrs[attr]
            except:
                print('{} already there'.format(obj.type))

        print('going to look for the following objects inside {}: \n{}'.format(group, [obj.type for obj in next_step_objs]))
        if len(next_step_objs) > 0:
            for r in range(src_dset.shape[0]):
                row = src_dset[r, ...]
                msg_id = row[13:21]
                msg_id.dtype = 'u8'
                for obj in next_step_objs:
                    if msg_id == obj.msg_id:
                        # print('Found {} at index {}'.format(obj.type, r))
                        F[obj.type]['raw_data'].resize(F[obj.type]['raw_data'].shape[0] + 1, axis=0)
                        # print('dset reshaped to {}'.format(F[obj.type]['raw_data'].shape))
                        F[obj.type]['raw_data'][-1:, ...] = row


class Player:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()

    def __init__(self, filename, group):
        self._frame_idx = mp.Value('i')
        self._frame_idx.value = 0
        self._is_ready = mp.Value('i')
        self._is_ready.value = 0
        self._str_id = group
        self._filename = filename
        self._group = group
        self._file = None
        self._dset = None

        file = h5py.File(filename, 'r')
        dset = file[group]['raw_data']

        self._dtype = dset.attrs['dtype']
        self._shape = dset.attrs['shape']

        self.params = dict()
        for key in dset.attrs.keys():
            self.params[key] = dset.attrs[key]
        self._n_frames = dset.shape[0]
        file.close()

    def open(self):
        self._file = h5py.File(self._filename, 'r')
        self._dset = self._file[self._group]['raw_data']
        self._is_ready.value = 1

    def display_params(self):
        """Prints all the parameters of an object"""
        print('\n\n{} PARAMETERS:'.format(self.str_id))
        for k in self.params.keys():
            print(k + ':', self.params[k])
        print('\n\n')

    @property
    def n_frames(self):
        return self._n_frames

    @property
    def is_ready(self):
        return bool(self._is_ready.value)

    @property
    def str_id(self):
        return self._str_id

    @property
    def frame_idx(self):
        return self._frame_idx.value

    @frame_idx.setter
    def frame_idx(self, value):
        self._frame_idx.value = value

    def next(self):
        # Increase the frame counter by 1
        self._frame_idx.value += 1

    def get_data(self, return_ts=False, auto_increment=False):
        try:
            row = np.copy(self._dset[self.frame_idx, ...])
        except IndexError:
            row = np.zeros((int(np.prod(self._shape)+self.params['header_len']),), dtype='B')
            ts = row[:8]
            ts.dtype = 'd'
            ts[:] = self.params['perf_counter']
            if self.params['type'] == 'vector can':
                row[self.params['header_len']] = 64

        if auto_increment:
            self.next()

        return get_data_by_type(self, row, return_ts)


class Opener(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # always remember to close the file
        self._required.value = 0

    def __init__(self, filename, autorun=False, parse=None, pause=False):
        """Replay an .h5 file.

        Keyword arguments:
            filename -- the name of the file you want to open.
            autorun -- automatically start the walk through the file.
            parse -- objects that have been transmitted on the CAN bus. Arrange them as [('cangroup, [list_of_objects])]

        This class can be treated as a dictionary which keys are the str_ids of the sensors you are using,
        and which items have the method .get_data() which returns different objects accordingly to the sensor itself.
        It also implements __enter__ and __exit__ methods so that you can use it in a with statement.
        """
        if parse is not None:
            if isinstance(parse, tuple):
                canparser(filename, parse[0], parse[1])
            elif isinstance(parse, list):
                for line in parse:
                    canparser(filename, line[0], line[1])
            else:
                print('Warning: invalid canparser arguments')
        with h5py.File(filename, 'r') as file:

            self._group_list = list(file)

            self._repr_speed = mp.Value('d')
            self._repr_speed.value = 1

            self._end_of_dataset = mp.Value('i')
            self._end_of_dataset.value = 0

            self._time = mp.Value('d')
            self._time.value = 0

            self._frame_idx = mp.Value('i')
            self._frame_idx.value = 0

            self._required = mp.Value('i')
            self._required.value = 1

            self._cursor_moved = mp.Value('i')
            self._cursor_moved.value = 0

            self._cursor_ready = mp.Value('i')
            self._cursor_ready.value = 0

            self._pause = mp.Value('i')
            self._pause.value = pause

            # This table will be [[timestamp, str_id(*), frame index]]
            # (*) it will actually be the index relative to group list
            self._timestamp_table = np.empty((0, 3), dtype='d')
            utc_ref = 2**32

            self._populated_list = []

            for g, i in zip(self._group_list, range(len(self._group_list))):
                self[g] = Player(filename, g)
                self._populated_list.append(self[g].n_frames)
                if self[g].params['time_ref'] < utc_ref:
                    utc_ref = self[g].params['time_ref']
                    try:
                        on_time_at_utc_ref = self[g].params['perf_counter']
                    except KeyError:
                        pass    # dirty fix because I forgot to put perf counter on it......
                        # it is enough to have one in the whole h5
                # populate the table
                timestamp_vector = file[g]['raw_data'][:, :8]
                print('str_id: {}, n_frames:{}, type: {}'.format(g,
                                                                 file[g]['raw_data'].shape[0],
                                                                 file[g]['raw_data'].attrs['type']))
                timestamp_vector.dtype = 'd'

                device_vector = np.empty((timestamp_vector.shape[0], 3), dtype='d')
                device_vector[:, 0] = timestamp_vector.reshape(-1)
                device_vector[:, 1] = i
                device_vector[:, 2] = np.arange(device_vector.shape[0])
                self._timestamp_table = np.concatenate((self._timestamp_table, device_vector))

            # sort the table according to the timestamp
            self._timestamp_table = np.take(self._timestamp_table,
                                            np.argsort(self._timestamp_table, axis=0)[:, 0], axis=0)
            min_time = self._timestamp_table[0, 0]
            self._timestamp_table[:, 0] -= min_time

        self.utc_offset = utc_ref-on_time_at_utc_ref+min_time

        if autorun:
            p = mp.Process(target=self._run)
            p.start()

        for g in self._group_list:
            self[g].open()

    @property
    def duration(self):
        return self._timestamp_table[-1, 0] - self._timestamp_table[0, 0]

    @property
    def time(self):
        return self._time.value

    @property
    def cursor_moved(self):
        return bool(self._cursor_moved.value)

    @cursor_moved.setter
    def cursor_moved(self, value):
        if value:
            self._cursor_moved.value = 1
        else:
            self._cursor_moved.value = 0
        while self._cursor_moved.value == 1:
            time.sleep(0.01)

    @time.setter
    def time(self, value):
        """
        Set the current file time.
        """
        if self._frame_idx.value+1 < self._timestamp_table.shape[0]:
            # If you move the time "to the future" wrt current time, it forces to take the next frame for every
            if value > self._time.value:
                self.idx = np.argmin(np.absolute(self._timestamp_table[self._frame_idx.value+1:, 0] - value))+self._frame_idx.value
            elif value > self._time.value:
                self.idx = np.argmin(np.absolute(self._timestamp_table[:self._frame_idx.value+1, 0] - value))
            ts, dev, frame_idx = self._timestamp_table[self.idx, :]
            self._time.value = ts

            self.cursor_moved = 1
        else:
            print('Reach the end of the dataset')
            self._end_of_dataset.value = True
            self.pause = True
            self.cursor_moved = False

    @property
    def eof(self):
        return bool(self._end_of_dataset.value)

    @property
    def repr_speed(self):
        return self._repr_speed.value

    @repr_speed.setter
    def repr_speed(self, value):
        self._repr_speed.value = max(0, value)

    @property
    def pause(self):
        return bool(self._pause.value)

    @pause.setter
    def pause(self, value):
        if value:
            self._pause.value = 1
        else:
            self._pause.value = 0

    @property
    def idx(self):
        return self._frame_idx.value

    @idx.setter
    def idx(self, value):
        self._frame_idx.value = value

    def _run(self):
        while True:

            # If pause is set, stay there and check if it is still required
            while self.pause and self._required.value and not self.cursor_moved:
                time.sleep(0.01)

            if not self._required.value:
                print('Stopping player.')
                return
            else:
                # If required, do the followings
                ts, dev, frame_idx = self._timestamp_table[self.idx, :]
                self._time.value = ts
                self[self._group_list[int(dev)]].frame_idx = int(frame_idx)

                if not self.cursor_moved:
                    if not (self.idx + 1) >= self._timestamp_table.shape[0]:
                        precise_sleep(min(1, (self._timestamp_table[self.idx+1, 0]-ts)/self.repr_speed))
                else:
                    # When moving the cursor, update all devices to the closest past frame
                    for k in range(len(self._group_list)):
                        if self._populated_list[k]:
                            # this code is executed only if there are frames available in a specified device.
                            dev = -1
                            displacement = 0

                            # You might be trying to go beyond the beginning of the file, so we add an if.
                            while dev != k:
                                if (self.idx - displacement) < self._timestamp_table.shape[0]:
                                    ts, dev, frame_idx = self._timestamp_table[self.idx - displacement, :]
                                    displacement += 1
                                else:
                                    ts, dev, frame_idx = self._timestamp_table[self.idx - displacement, :]
                                    displacement -= 1
                            self[self._group_list[k]].frame_idx = int(frame_idx)

                    self._cursor_moved.value = 0
                self.idx += 1
                if self.idx >= self._timestamp_table.shape[0]:
                    print('PLAYER: reached the end of the dataset')
                    self._end_of_dataset.value = True
                    self.pause = True
                    self.cursor_moved = False
