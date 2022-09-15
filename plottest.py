from process.plotting import PolarProjection
import numpy as np
import time

if __name__ == '__main__':
    pro = PolarProjection(np.arange(20)/10+1, np.arange(180)/180*np.pi,title='CICCIO', db=False)

    while pro.is_alive:
        ev = pro.get_event()
        if ev is not None:
            print(ev)
        mat = np.random.rand(20, 180)
        mat[::2, :] = 0
        mat[:, 1::2] = 1
        pro.update(mat)
        time.sleep(0.01)
        pro.addon_scatter[:]=np.random.rand(3, 2)+1