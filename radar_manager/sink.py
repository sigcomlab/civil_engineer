from multiprocessing import Process, Value, current_process
from numpy import prod, dtype
from qtplots import MessageBox
import logging
import h5py
import time


def precise_sleep(t):
    # If the time to sleep is not that small, we can use the less precise time.sleep()
    if t < 0.01:
        target_time = time.perf_counter() + t
        while time.perf_counter() < target_time:
            pass
    else:
        time.sleep(t)


TIMESTAMP_LEN = 8
COUNTER_LEN = 4


class HdfSink:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if exc_type:
            print('HdfSink:')
            print('exc_type: {}'.format(exc_type))
            print('exc_value: {}'.format(exc_val))
            print('exc_traceback: {}'.format(exc_tb))

    def __init__(self, filename, object_list, autostart=True, display_bufferstate=True):
        """
        This object stores data coming from objects inside object_list. If not autostarted, it must be started later
        """
        self._filename = filename
        self._required = Value('i')
        self._required.value = 0
        # self._started = False   # Flag to see if a call to start() has been made
        self._object_list = object_list     # list of objects to monitor
        self._storage = None    # hdf file opened
        self._grp = None    # group(s) inside hdf file
        self._display_bufferstate_flag = display_bufferstate
        self._warning_triggered_50 = False  # flags to avoid flooding the log in case of buffer too full
        self._warning_triggered_80 = False

        if filename == () or filename == '':
            logging.warning('HdfSink: no filename specified so nothing will be saved')
            self._valid = False     # flag to see if HdfSink is working properly or not
        else:
            self._valid = True
            if autostart:
                self.start()

    def start(self, filename=None):
        """
        Starts to save with the specified filename. If filename is None, use the default filename instead
        """
        if filename is not None:
            if filename == () or filename == '':
                logging.warning('HdfSink: no filename specified so nothing will be saved')
                self._valid = False  # flag to see if HdfSink is working properly or not
                return
            else:
                self._valid = True
            self._filename = filename
        if not self._required.value and self._valid:   # If not yet started and still valid:
            self._required.value = 1
            proc = Process(target=self._run)    # ... starts the process.
            proc.start()
        elif self._required.value:
            logging.warning('HdfSink: already started')
        elif not self._valid:
            logging.warning('HdfSink compulsory closed, non valid to start')
        else:
            logging.error('HdfSink: unforeseen situation')

    def _run(self):
        logging.info('SINK started with PID {} - filename = "{}"'.format(current_process().pid, self._filename))

        with h5py.File(self._filename, 'a', libver='latest') as self._storage:
            buf_dict = {}
            try:
                for new_object in self._object_list:
                    self._add_object(new_object)  # Adds the object to the monitored ones
                    buf_dict[new_object.str_id] = (0, 0, 0)   # let's init the first buffer occupancy at 0

                if self._display_bufferstate_flag:
                    self._msgbox = MessageBox(title='BUFFER STATES', refresh_ms=500)

                while self._required.value:
                    for existing_object in self._object_list:
                        # Scans sensors in polling
                        # What follows are to prevent the sink to call the flush_buffer() too often
                        if len(self._object_list) == 1:
                            precise_sleep(0.2)

                        str_id = existing_object.str_id

                        if self._display_bufferstate_flag:
                            # If we want to see the amount of data in the buffer, put the values in a dictionary
                            # which key is the str_id of the object.
                            buf_dict[str_id] = (existing_object.scb.status,
                                                existing_object.scb.n_elems,
                                                existing_object.scb.num_written)

                        hyperframe = existing_object.flush_buffer()     # get all available data from device

                        if hyperframe is not None:
                            self._put_on_disk((str_id, hyperframe))     # put all the data into the hdf5 file

                    if self._display_bufferstate_flag:
                        self._update_bufferstate(buf_dict)
            except KeyboardInterrupt:
                pass
        if self._display_bufferstate_flag:
            self._msgbox.kill()
        logging.debug('SINK {} stopped'.format(current_process().pid))

    def stop(self):
        if self._required.value:
            # do this only if the object has already been start()ed
            self._required.value = 0
        else:
            logging.info('HdfSink not running: nothing to stop')

    def _add_object(self, new_object, compression=None):
        while not new_object.is_ready:
            # wait for the object to be ready
            time.sleep(0.1)
        print(new_object.str_id, 'is ready')

        # check if we want to compress the dataset
        if 'compression' in new_object.params.keys() \
                and new_object.params['compression'] != 'none' \
                and new_object.params['compression'] is not None:
            compression = new_object.params['compression']

        # make sure we have the shape as attribute
        if 'shape' not in new_object.params.keys():
            new_object.params['shape'] = new_object.shape

        if 'dtype' not in new_object.params.keys():
            logging.warning('SINK: Data type not specified for ', new_object.str_id, ': using int16')
            new_object.params['dtype'] = 'i2'

        new_object.params['version'] = '0.3'    # insert here the add_object version (for evolution references)

        str_id = new_object.str_id
        # The total amount of bytes to be stored into the h5 is given by
        # || TIMESTAMP | COUNTER |  R A W _ D A T A  ||
        if 'header_len' in new_object.params.keys():
            d_shape = \
                (new_object.params['header_len'] + prod(new_object.shape) * dtype(new_object.params['dtype']).itemsize,)
        else:
            d_shape = \
                (TIMESTAMP_LEN + COUNTER_LEN + prod(new_object.shape) * dtype(new_object.params['dtype']).itemsize,)
        logging.debug("SINK: added '{}' with shape {} and buffer depth {}".format(str_id, new_object.params['shape'],
                                                                                  new_object.scb.shape[0]))
        h5_chunk = ((1,) + d_shape)     # see h5py documentation if interested, otherwise live it as is

        # if the object has not been added yet, create its group and dataset inside the file
        if str_id not in self._storage.keys():
            self._grp = self._storage.create_group(str_id)
            data_dataset = self._grp.create_dataset('raw_data',
                                                    shape=((0,) + d_shape),
                                                    maxshape=((None,) + d_shape),
                                                    chunks=h5_chunk,
                                                    dtype='B',
                                                    compression=compression)

            # add all its attributes (i.e bandwidth, ts...) to the file
            attributes = new_object.params
            for key in attributes:
                data_dataset.attrs[key] = attributes[key]

        # Wipe the buffer from all the old data
        new_object.wipe_buffer()
        logging.debug('SINK: buffer wiped for {}'.format(str_id))

    def _put_on_disk(self, id_hyperframe):
        """
        Write the data to the file
        :param id_hyperframe: tuple (str_id, hyperframe)
        :return: nothing
        """
        (str_id, hyperframe) = id_hyperframe
        radar_raw_dset = self._storage[str_id]['raw_data']
        radar_raw_dset.resize(radar_raw_dset.shape[0] + hyperframe.shape[0], axis=0)
        radar_raw_dset[-hyperframe.shape[0]:, ...] = hyperframe

    def _update_bufferstate(self, bufdict):
        """
        Update the message box text
        """
        txt = ''
        for (str_id, state) in bufdict.items():
            txt += '{} - elems: {}, tot written: {}, percentage fill: {}% \n'.format(str_id,
                                                                                     state[0],
                                                                                     state[2],
                                                                                     state[1])
            # Small control on buffer state and eventual logging
            if state[1] > 50 and not self._warning_triggered_50:
                self._warning_triggered_50 = True
                logging.warning("SINK: '{}' buffer is {}% full".format(str_id, state[1]))
            elif state[1] > 80 and not self._warning_triggered_80:
                self._warning_triggered_80 = True
                logging.critical("SINK: '{}' buffer is {}% full. Frame loss will occur soon".format(str_id, state[1]))
            elif state[1] <= 50 and self._warning_triggered_50:
                # reset the flags if the buffer is safe again
                self._warning_triggered_50 = False
                logging.warning("SINK: '{}' buffer is now safe again".format(str_id))
            elif state[1] <= 80 and self._warning_triggered_80:
                self._warning_triggered_80 = False
                logging.warning("SINK: '{}' buffer is less than 80% but still dangerous".format(str_id))

        self._msgbox.text = txt
