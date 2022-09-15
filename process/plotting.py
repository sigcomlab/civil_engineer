# noinspection PyUnresolvedReferences
import time
from multiprocessing import Process, Queue
from numpy import meshgrid, absolute, log10, amin, amax, any, iscomplex, c_
from pybuf import HiddenMemory
import matplotlib.pyplot as plt
from queue import Empty
from time import sleep
import warnings
from coords import Coords3D

"""
tested on python 3.9, matplotlib version = 3.5.3. BE CARFUL! ON 3.5.0 IT DOES NOT WORK!!!
"""


class PolarProjection:
    def __init__(self, range_scale, angular_scale, title='', db=True, cm='viridis', colorbar=None):
        """
        Make a polar plot with arbitrary matrix, colormap and scales
        :param range_scale: range axis scale.
        :param angular_scale: angle axis scale.
        :param title: Title of the plot to be displayed
        :param db: Do the logarithm of the incoming data. True or False.
        :param cm: colormap to be used. See matplotlib colormaps for options.
        :param colorbar: Add the colorbar next to the plot with specified extremes like [0, 5]
        """
        shared_memory_size = 4096 * 512

        self.db = db
        self.cm = cm
        if colorbar is None:
            self._colorbar = [0, 1]
        else:
            self._colorbar = colorbar
        self._cmd_queue = Queue()
        self._ans_queue = Queue()

        self._array = HiddenMemory(dtype='d', n_elems=shared_memory_size)
        self._range_scale = range_scale
        self._angular_scale = angular_scale
        self._array.shape = (self._range_scale.shape[0], self._angular_scale.shape[0])

        self.addon_scatter = HiddenMemory(dtype='d', n_elems=6)
        self.addon_scatter.shape = (3, 2)

        self.title = title
        self.proc = Process(target=self._run, daemon=True)
        self.proc.start()

    def update(self, matrix_to_plot):
        """
        Call this method to update the figure
        :param matrix_to_plot: matrix to plot with dimensions (range, theta)
        :return: nothing
        """
        if len(matrix_to_plot.shape) == 3:
            matrix_to_plot = matrix_to_plot[..., -1]
        if any(iscomplex(matrix_to_plot)):
            matrix_to_plot = absolute(matrix_to_plot)
        if self.db:
            matrix_to_plot = log10(absolute(matrix_to_plot))
        if self._colorbar == [0, 1]:
            matrix_to_plot -= amin(matrix_to_plot)
            matrix_to_plot /= (amax(matrix_to_plot))+0.001
        self._array[...] = matrix_to_plot

    @property
    def is_alive(self):
        """
        Call this method to check if the figure is still present on the screen
        :return: True or False
        """
        return self.proc.is_alive()

    def _on_key(self, event):
        self._ans_queue.put(('k', event.key))

    def _on_click(self, event):
        self._ans_queue.put(('m', event.xdata, event.ydata))

    def get_event(self):
        """
        check if user has interacted with the plot by clicking on it or pressing keys
        :return: key pressed or Coord object in case of mouse click
        """
        try:
            event = self._ans_queue.get(block=False)
        except Empty:
            return None
        if event[0] == 'k':
            return event[0], event[1]
        elif event[0] == 'm' and event[1] is not None:
            point = Coords3D((1,))
            point.azimuth = event[1]    # radiants.
            point.range = event[2]
            return event[0], point

    def _run(self):
        required = True
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_thetalim([self._angular_scale[0], self._angular_scale[-1]])
        ax.set_axisbelow(False)
        ax.set_theta_zero_location("N")
        # ax.set_title(self.title, family='serif', weight='bold', size='x-large')

        angular_ax, range_ax = meshgrid(self._angular_scale, self._range_scale)

        im = ax.pcolormesh(angular_ax, range_ax, self._array[:, :],
                           vmin=self._colorbar[0],
                           vmax=self._colorbar[1],
                           cmap=plt.get_cmap(self.cm), zorder=1)

        if self._colorbar != [0, 1]:
            fig.colorbar(im, ax=ax)

        sca = ax.scatter(self.addon_scatter[:, 0], self.addon_scatter[:, 1], c=['r', 'g', 'b'], zorder=2)

        kp = im.figure.canvas.mpl_connect('key_press_event', lambda event: self._on_key(event, ))
        bp = im.figure.canvas.mpl_connect('button_press_event', lambda event: self._on_click(event, ))
        fig.canvas.draw()   # This one initialize something inside matplotlib
        plt.show(block=False)

        while required:

            matrix_to_plot = self._array[...]
            im.set_array(matrix_to_plot[:, :].flatten())

            try:
                ret = self._cmd_queue.get(block=False)
                cmd = ret[0]
                if cmd == 'title':
                    ax.set_title(ret[1], family='serif', weight='bold', size='x-large')
                else:
                    warnings.warn('invalid command', RuntimeWarning)
            except Empty:
                sleep(0.05)

            sca.set_offsets(c_[self.addon_scatter[:, 0], self.addon_scatter[:, 1]])
            ax.draw_artist(ax.collections[0])

            # fig.canvas.draw()    # Probably, fig.canvas.draw() is more portable/compatible than draw_artist.

            fig.canvas.flush_events()
            # plt.pause(0.01)

            if not plt.get_fignums():
                im.figure.canvas.mpl_disconnect(kp)
                im.figure.canvas.mpl_disconnect(bp)
                required = False

    def update_title(self, new_title):
        self._cmd_queue.put(('title', new_title))
