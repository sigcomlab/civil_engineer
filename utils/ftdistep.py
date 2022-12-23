from pyftdi.gpio import GpioAsyncController
import time


class FtdiStepper:
    def __init__(self):
        device = 'ftdi://ftdi/1'

        self.gpioa = GpioAsyncController()

        self.steppin = 2
        self.dirpin = 3
        self.enablepin = 4

        self.STEPCOUNT = 0

        dirmask = 0x0 | 1 << self.steppin
        dirmask |= 1 << self.dirpin
        dirmask |= 1 << self.enablepin

        self.gpioa.configure(device, direction=dirmask)

        self.register = 0
        self.step_per_mm = 50

    def disable(self):
        self.register = self.register | 1 << self.enablepin
        self.gpioa.write(self.register)

    def enable(self):
        self.register = self.register & (0xff ^ 1 << self.enablepin)
        self.gpioa.write(self.register)

    def dosteps(self, nsteps, direction=0, period=0.01):

        if nsteps < 0:
            direction = 1
            nsteps *= -1
        for _ in range(nsteps):

            self.register = self.register & (0xff ^ (1 << self.dirpin | 1 << self.steppin))    # clear stuff
            self.register = self.register | direction << self.dirpin
            self.register = self.register | 1 << self.steppin

            self.gpioa.write(self.register)
            time.sleep(period/2)
            self.register = self.register & ((1 << self.steppin) ^ 0xff)
            self.gpioa.write(self.register)
            time.sleep(period / 2)
            self.STEPCOUNT -= (2*direction)-1
