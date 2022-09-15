from numpy import frombuffer, prod, \
    int8, uint8, int16, uint16, int32, uint32, float32, float64
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import Tk, Button, OptionMenu, StringVar, messagebox, Scale, HORIZONTAL, \
                    Frame, Label, Entry, BOTH, TOP, LEFT, BOTTOM, TclError
from pyqtgraph.Qt import QtCore
from struct import unpack_from
import ctypes

import platform
import json
import warnings
import h5py
import time
import sys
import os

root_dir = os.path.dirname(__file__).split('utils')[0]

DTYPES = {
    0: int8,
    1: uint8,
    2: int16,
    3: uint16,
    4: int32,
    5: uint32,
    6: float32,
    7: float64,
}

ASCII_RS = '\u001e'
ASCII_US = '\u001f'

if platform.system() == 'Linux':
    perf_counter = time.perf_counter
    perf_counter_ns = time.perf_counter_ns
else:
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    kernel32.QueryPerformanceFrequency.argtypes = (
        wintypes.PLARGE_INTEGER,)  # lpFrequency

    kernel32.QueryPerformanceCounter.argtypes = (
        wintypes.PLARGE_INTEGER,)  # lpPerformanceCount

    _qpc_frequency = wintypes.LARGE_INTEGER()
    if not kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_frequency)):
        raise ctypes.WinError(ctypes.get_last_error())
    _qpc_frequency = _qpc_frequency.value


    def perf_counter_ns():
        """perf_counter_ns() -> int

        Performance counter for benchmarking as nanoseconds.
        """
        count = wintypes.LARGE_INTEGER()
        if not kernel32.QueryPerformanceCounter(ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (count.value * 10**9) // _qpc_frequency


    def perf_counter():
        """perf_counter() -> float

        Performance counter for benchmarking.
        """
        count = wintypes.LARGE_INTEGER()
        if not kernel32.QueryPerformanceCounter(ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        return count.value / _qpc_frequency


def precise_sleep(t):
    # If the time to sleep is not that small, we can use the less precise time.sleep()
    if t < 0.01:
        target_time = perf_counter() + t
        while perf_counter() < target_time:
            pass
    else:
        time.sleep(t)


def split(a, n):
    k, m = divmod(len(a), n)
    return[a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


class LabelInput:
    def __init__(self):
        self.root = Tk()
        self.root.title("LabelInput by UNIMORE SIGCOM Lab")
        # img = Image("photo", file="logo_2.gif")
        # self.root.tk.call('wm', 'iconphoto', self.root._w, img)
        self._required = False

        topframe = Frame(self.root)
        lab = Label(topframe, text="IP: ")
        self._e1 = Entry(topframe, textvariable=StringVar(self.root, value=''))
        butt = Button(topframe, text="OK", command=self._ok_button)
        lab.pack(padx=0, pady=0, side=LEFT)
        self._e1.pack(padx=0, pady=0, side=LEFT, fill=BOTH)
        butt.pack(padx=0, pady=0, side=BOTTOM, fill=BOTH)

        topframe.pack(side=TOP, fill=BOTH)

        self.root.update()

    def input_text(self):
        try:
            self.root.update()
            if self._required:
                self._required = False
                return self._e1.get()
        except TclError:
            pass

    def _ok_button(self):
        self._required = True


class GroupPicker:
    """
    List all the groups (aka Radars or Lidars) in a hdf file and pick one
    """
    def __init__(self, filename):
        self.root = Tk()
        self.root.title("Choose the radar")
        self.root.geometry("200x50")
        self.group = 'dummy'
        options = hdf_prober(filename)
        self.variable = StringVar(self.root)
        self.variable.set(options[0])
        w = OptionMenu(self.root, self.variable, *options)
        w.pack()
        button = Button(self.root, text='OK', command=self._quit)
        button.pack()
        self.root.mainloop()

    def _quit(self):
        self.group = self.variable.get()
        self.root.quit()
        self.root.destroy()


class FrameSaver:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._storage.close()
        if exc_type:
            print('FrameSaver:')
            print('exc_type: {}'.format(exc_type))
            print('exc_value: {}'.format(exc_val))
            print('exc_traceback: {}'.format(exc_tb))

    def __init__(self, filename, radar, label=None, picoflex=None):
        self.filename = filename
        self.str_id = radar.str_id
        self._storage = h5py.File(self.filename, 'a', libver='latest')
        self._grp = None

        self.label = label
        self.picoflex = picoflex
        if picoflex is None:
            self.picoshape = (0,)
        else:
            self.picoshape = picoflex.shape
        self._add_object(radar)

    def append(self, hyperframe, label=None, picopicture=None):
        radar_raw_dset = self._storage[self.str_id]['raw_data']
        radar_raw_dset.resize(radar_raw_dset.shape[0] + 1, axis=0)
        radar_raw_dset[-1, ...] = hyperframe

        if picopicture is not None:
            radar_pic_dset = self._storage[self.str_id]['picoflexx']
            radar_pic_dset.resize(radar_pic_dset.shape[0] + 1, axis=0)
            radar_pic_dset[-1, ...] = picopicture

        if label is not None:
            radar_lab_dset = self._storage[self.str_id]['label']
            radar_lab_dset.resize(radar_lab_dset.shape[0] + 1, axis=0)
            radar_lab_dset[-1, ...] = int8(int(label, 2))

    def _add_object(self, new_object):
        while not new_object.is_ready:
            time.sleep(1e-2)

        str_id = new_object.str_id
        d_shape = new_object.shape

        print('added  "', str_id, '"  with shape', d_shape)
        h5_chunk = ((1,) + d_shape)
        if str_id not in self._storage.keys():
            self._grp = self._storage.create_group(str_id)
            data_dataset = self._grp.create_dataset('raw_data',
                                                    shape=((0,) + d_shape),
                                                    maxshape=((None,) + d_shape),
                                                    chunks=h5_chunk,
                                                    dtype='int16')
            if self.picoflex is not None:
                self._grp.create_dataset('picoflexx',
                                         shape=((0,) + self.picoshape),
                                         maxshape=((None,) + self.picoshape),
                                         dtype='float32')
            if self.label is not None:
                self._grp.create_dataset('label',
                                         shape=(0, 1),
                                         maxshape=(None, 1),
                                         dtype='int8')

            attributes = new_object.params
            for key in attributes:
                data_dataset.attrs[key] = attributes[key]


def askopenfile(folder='hdf_files', extension='*.h5', description=''):
    root = Tk()
    root.withdraw()

    if extension == '*.h5':
        description = "h5 files"
    elif extension == '*.npy':
        description = 'numpy array'
    filename = askopenfilename(initialdir=os.path.join(root_dir, folder),
                               filetypes=((description, extension),))
    root.quit()
    root.destroy()
    return filename

def askopenfolder():
    root = Tk()
    root.withdraw()

    path = askopenfolder()
    root.quit()
    root.destroy()
    return path


def message_box(message):
    root = Tk()
    root.withdraw()
    state = messagebox.askyesno(title='UNIMORE SIGCOMM Lab', message=message)
    root.quit()
    root.destroy()
    return state


def hdf_prober(filename):
    """
    Returns a list of the group (i.e. radars) stored inside an hdf5 file
    """
    if filename == () or filename == '':
        warnings.warn('No filename provided', RuntimeWarning)
        sys.exit()
    with h5py.File(filename, 'r', libver='latest') as file:
        group_list = list(file.keys())
    return group_list


def asksavefile(folder='hdf_files', initialfile='', extension='*.h5', description=''):
    root = Tk()
    root.withdraw()

    if description == '':
        if extension == '*.h5':
            description = "h5 files"
        elif extension == '*.npy':
            description = 'numpy array'

    filename = asksaveasfilename(initialdir=os.path.join(root_dir, folder), initialfile=initialfile,
                                 defaultextension=extension, filetypes=((description, extension), ))
    root.quit()
    root.destroy()
    return filename


def intelligent_sleep(dev_list, plot_to_handle=None, t_res=0.01):
    last_list = [dev.is_last_read for dev in dev_list]
    precise_sleep(t_res/2)
    while prod(last_list):
        last_list = [dev.is_last_read for dev in dev_list]
        if plot_to_handle is not None:
            plotalive_list = [plot.is_alive for plot in plot_to_handle]
            if not prod(plotalive_list):
                return False
        # print(last_list, prod(last_list))
        precise_sleep(t_res)
    return True

