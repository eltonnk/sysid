import numpy as np
# from scipy import signal
from matplotlib import pyplot as plt
import pathlib
import plant_util as util

# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# path = pathlib.Path('figs')
# path.mkdir(exist_ok=True)

for i in range(1, 6):
    sd = util.load_sensor_data(i)
    sd.plot()

plt.show()