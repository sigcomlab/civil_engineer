from radar_manager.velodyne_manager import LidarManager
from radar_manager.sink import HdfSink
from utils.utils import askopenfile, asksavefile
from qtplots import Generic3DScatterItem, Plot3D, MessageBox
from utils.utils import intelligent_sleep
from multiprocessing import set_start_method
import shutil
import time

replay = False
sf = 1


if __name__ == '__main__':
    set_start_method('spawn')

    if replay:
        filename = askopenfile()
    else:
        filename = 'dummyfile'

    MSG = MessageBox(title='Live monitor')

    with LidarManager('lidar_l', port=2369) as L, \
            HdfSink(filename, [L], autostart=False) as S:

        while MSG.is_alive:
            # Check the amount of free space on the disk
            total, used, free = shutil.disk_usage("/")
            free //= 1024*1024*1024
            if not replay:
                # Eventually start the sink
                filename = asksavefile()
                S.start(filename)

            # create the 3D plot window
            plot3d = Plot3D({'velo': Generic3DScatterItem(n_points=L.params['points'],
                                                          color='g')}, title='LIDAR scatter plot')

            tstart = time.time()
            while intelligent_sleep([L], [plot3d, MSG], t_res=0.01):
                t0 = time.time()

                lidar_points = L.get_data().rotate(0, axis='z').translate([0.05, -2, 0])
                plot3d['velo'].xyz[:len(lidar_points)] = lidar_points.cartesian_coords[:len(lidar_points)]

                t_tot = time.time() - t0

                txt = 'free space: {} GB \n' \
                      'process time: {:.3f} s \n'\
                      'tot time {}'.format(free, t_tot, int(time.time()-tstart))
                MSG.text= txt # Update the text in the messagebox

            time.sleep(1)
