from multiprocessing import Process, Queue, Lock, Value
from pybuf import CircularBuffer
from radar_manager.get_data import get_data_by_type
import warnings
import time


class PhyObject:
    def __init__(self, str_id, dtype='d', mem_footprint=4096 * 128 * 196 + 256, ext_obj=None):
        """
        Initialize PhyObject (which properties are inherited by  almost everything)
        :param str_id: unique identifier for this object
        :param dtype: data type this object is going to generate
        :param mem_footprint: number of element in its buffer
        """
        self._cmd_queue = Queue()
        self._ans_queue = Queue()
        self._external_lock = Lock()
        self.ext_obj = ext_obj

        self._is_last_read = Value('i')
        self._is_last_read.value = 1

        # This value is set to True only when the object is ready to provide data
        self._is_ready = Value('i')
        self._is_ready.value = 0

        self._pause = Value('i')
        self._pause.value = 0

        self.str_id = str_id
        self._params = None

        self._shared_memory_size = mem_footprint  # this is the number of elements we are going to bufferize
        self.scb = CircularBuffer(self._shared_memory_size, dtype=dtype)

        # max number of elements that are read simultaneously when flushing the circular buffer
        self._hyperframe_buffer = 32

        self.started = False

        # Number of element read
        self.counter = Value('I')
        self.counter.value = 0

    @property
    def is_last_read(self):
        return bool(self._is_last_read.value)

    @is_last_read.setter
    def is_last_read(self, value):
        self._is_last_read.value = int(value)

    @property
    def is_ready(self):
        """ Returns True when the object is initialized and ready to send data"""
        return bool(self._is_ready.value)

    def start(self, method=None):
        """ Launch the method as separate process """
        if method is None:
            # Choose the method to start as process
            method = self._run
        if not self.started:
            proc = Process(target=method, daemon=True)
            proc.start()
            self.started = True
            # At this point the child process is started but not ready yet
            while not self._is_ready:
                time.sleep(0.2)
        else:
            warnings.warn('Object already started: nothing to do')

    def display_params(self):
        """Prints all the parameters of an object"""
        print('\n\n{} PARAMETERS:'.format(self.str_id))
        for k in self.params.keys():
            print(k + ':', self.params[k])
        print('\n\n')

    def stop(self):
        """Send the stop command to the child process"""
        self._cmd_queue.put(('stop',))

    @property
    def shape(self):
        """
        This function returns the shape of the data you are going to get. It may be useful for initializations.
        :return: shape in the form (time_samples, virtual_channels, doppler_frames)
        """
        if 'shape' in self.params.keys():
            d_shape = tuple(self.params['shape'])
        else:
            d_shape = self.scb.shape[1:]
        return d_shape

    def get_data(self, return_ts=False):
        # in case of error, add return_counter=False in the arguments
        """
        Get the last available data from whatever sensor
        """
        row = self.scb.get_last()
        self.is_last_read = True

        return get_data_by_type(self, row, return_ts)

    def get_circular_buffer(self):
        """
        Get all the buffer with latest and older frames
        :return: all the array with flattened data
        """
        new_array = self.scb.read(self._hyperframe_buffer, update=False)
        return new_array

    @property
    def params(self):
        """
        get the dictionary of the object parameters
        :return: params dictionary
        """
        if self._params is None:
            with self._external_lock:
                self._cmd_queue.put(('params',))
            self._params = self._ans_queue.get(block=True, timeout=5)
        return self._params

    def flush_buffer(self):
        """
        return the data from the last call to this function. It is similar to get_circular_buffer except that the former
        always returns new data
        :return: data or None if no new data are available
        """
        new_array = self.scb.read(self._hyperframe_buffer, update=True)
        if new_array.size == 0:
            new_array = None
        return new_array

    def wipe_buffer(self):
        self.scb.wipe()

    def _run(self):
        """
        This is a dummy method that must be overridden in the superclass.
        :return: Throws a RuntimeError
        """

        raise RuntimeError('Wrong run method for {}'.format(self.str_id))
