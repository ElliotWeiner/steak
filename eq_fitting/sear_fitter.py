import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------------
# data
# ----------------------------------
raw_data = """
0 355 0
30 330 30
60 330 40
90 330 45
120 330 48
150 330 56
180 330 62
210 330 66
240 330 70
270 330 78
300 330 82
330 330 85
360 330 86
390 330 88
420 330 90
450 330 90
480 330 92
510 330 93
540 330 94
0 315 0
30 290 20
60 290 30
90 290 38
120 290 43
150 290 47
180 290 51
210 290 56
240 290 63
270 290 66
300 290 71
330 290 73
360 290 77
390 290 78
420 290 79
450 290 79
480 290 82
510 290 84
540 290 86
570 290 87
600 290 87
630 290 88
660 290 88
690 290 89
720 290 89
750 290 89
780 290 90
810 290 91
840 290 91
870 290 92
900 290 93
0 350 0
30 325 25
60 325 38
90 325 43
120 325 46
150 325 52
180 325 59
210 325 62
240 325 67
270 325 72
300 325 76
330 325 82
360 325 84
390 325 87
420 325 89
450 325 90
480 325 90
510 325 92
540 325 93
570 325 95
0 420 0
30 390 35
60 390 43
90 390 52
120 390 56
150 390 62
180 390 67
210 390 73
240 390 78
270 390 81
300 390 85
330 390 88
360 390 90
390 390 92
420 390 94
450 390 96
"""

data = np.array([list(map(float, line.split()))
                 for line in raw_data.strip().splitlines()])
time     = data[:,0]
pan_temp = data[:,1]
C_meas   = data[:,2]

# fixed Cmax
Cmax = 100.0

# ----------------------------------
# C(t, Tpan) = Cmax*(1 - exp(-a*(Tpan - b)*t))
# ----------------------------------
def model(inputs, a, b):
    t, T = inputs
    return Cmax * (1 - np.exp(-a*(T - b)*t))

# ----------------------------------
# fit
# ----------------------------------
p0 = [3e-5, 170]  # broad initial guess
popt0, _ = curve_fit(lambda inp, a, b: model(inp, a, b),
                     (time, pan_temp), C_meas, p0=p0)

# build +-50% bounds around that
lower_bounds = popt0 * 0.5
upper_bounds = popt0 * 1.5

# ----------------------------------
# refit with bounds
# ----------------------------------
popt, pcov = curve_fit(lambda inp, a, b: model(inp, a, b),
                       (time, pan_temp), C_meas,
                       p0=popt0,
                       bounds=(lower_bounds, upper_bounds),
                       maxfev=200000)

a_fit, b_fit = popt
print(f"a = {a_fit:.6e}")
print(f"b = {b_fit:.3f}")

# ----------------------------------
# plotting
# ----------------------------------
t_vals = np.linspace(time.min(), time.max(), 100)
T_vals = np.linspace(pan_temp.min(), pan_temp.max(), 100)
tt, TT = np.meshgrid(t_vals, T_vals)
ZZ = model((tt, TT), a_fit, b_fit)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(tt, TT, ZZ, cmap='viridis', alpha=0.6, edgecolor='none')

ax.scatter(time, pan_temp, C_meas, color='red', s=15, label='Data')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Pan Temp (°F)")
ax.set_zlabel("Color Index")
ax.set_title("Exponential Saturation Fit (Cmax=100) — 3D View")
ax.legend()

plt.tight_layout()
plt.show()
