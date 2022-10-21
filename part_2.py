"""MECH 412 project part 2 sample code for students.

James Forbes
2022/03/23
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
import control
import plant_util as util

# Functions
def circle(x_c, y_c, r):
    """Plot a circle at point (x_c, y_c) of radius r."""
    th = np.linspace(0, 2 * np.pi, 100)
    x = x_c + np.cos(th) * r
    y = y_c + np.sin(th) * r
    return x, y


def robust_nyq(P_nom, W2, wmin, wmax, N_w):
    """Robust Nyquist plot.
    Can be use to plot the nominal plant P(s) with `uncertainty'
    cirlces, or plot L(s) with `uncertainty circles'."""

    # Plot Nyquist plot, output if stable or not
    w_shared = np.logspace(wmin, wmax, N_w)
    # Call control.nyquist to get the count of the -1 point
    count_P_nom = control.nyquist_plot(P_nom, omega=w_shared, plot=False)

    # Set Nyquist plot up
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$Re$')
    ax.set_ylabel(r'$Im$')
    ax.plot(-1, 0, '+', color='C3')

    # Nominal plant magnitude and phase
    mag_P_nom, phase_P_nom, _ = control.bode(P_nom, w_shared, plot=False)
    Re_P_nom = mag_P_nom * np.cos(phase_P_nom)
    Im_P_nom = mag_P_nom * np.sin(phase_P_nom)

    # Plot Nyquist plot
    ax.plot(Re_P_nom, Im_P_nom, '-', color='C3')    
    # ax.plot(Re_P_nom, -Im_P_nom, '-', color='C3')

    number_of_circles = 50  # this can be changed
    w_circle = np.geomspace(10**wmin, 10**wmax, number_of_circles)
    mag_P_nom_W2, _, _ = control.bode(P_nom * W2, w_circle, plot=False)
    mag_P_nom, phase_P_nom, _ = control.bode(P_nom, w_circle, plot=False)
    Re_P_nom = mag_P_nom * np.cos(phase_P_nom)
    Im_P_nom = mag_P_nom * np.sin(phase_P_nom)
    for i in range(w_circle.size):
        x, y = circle(Re_P_nom[i], Im_P_nom[i], mag_P_nom_W2[i])
        ax.plot(x, y, color='C1', linewidth=0.75)

    return count_P_nom, fig, ax


if __name__ == '__main__':

    show_graphs = True
    investigate_ORHP = True

    # Plotting parameters
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif', size=14)
    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')


    # Common parameters
    # Laplace variable
    s = control.tf('s')

    # Bode plot frequency bounds and number of points
    N_w = 500
    w_shared = np.logspace(-3, 3, N_w)

    # Create nominal transfer function
    P_nom_plant = util.load_plant_from_file('good_plants/v2/P_nom_v2.json')
    P = P_nom_plant.delta_y_over_delta_u_c
    if investigate_ORHP:
        print(f'Nominal P = {P}\n')
        print(f'Zeros of Nominal P: {P.zeros()}')
        print(f'Poles of Nominal P: {P.poles()}')

    # W2, uncertainty weight
    W2_plant = util.load_plant_from_file('good_plants/v2/W2_v2.json')
    W2 = W2_plant.delta_y_over_delta_u_c

    mag_abs_W2, _, w = control.bode(W2, w_shared, plot=False)
    mag_dB_W2 = 20 * np.log10(mag_abs_W2)

    if show_graphs:
        fig, ax = plt.subplots()
        # fig.set_size_inches(8.5, 11, forward=True)
        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
        # Magnitude plot (dB).
        ax.semilogx(w, mag_dB_W2, '-', color='C0', label=r'$|W_2(j \omega)|$')
        ax.legend(loc='best')
        fig.tight_layout()
        # fig.savefig('.pdf')


    # W1, performance weight
    FS = 11.12e3

    gamma_r = 2/100
    omega_r = 1.2e0
    W1 = (1/gamma_r) * (1 / (s / omega_r + 1) )**2  # dummy value, you change

    mag_abs_W1, _, w = control.bode(W1, w_shared, plot=False)
    mag_dB_W1 = 20 * np.log10(mag_abs_W1)

    if show_graphs:
        fig, ax = plt.subplots()
        # fig.set_size_inches(8.5, 11, forward=True)
        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
        # Magnitude plot (dB).
        ax.semilogx(w, mag_dB_W1, '-', color='C0', label=r'$|W_1(j \omega)|$')
        ax.legend(loc='best')
        fig.tight_layout()
        # fig.savefig('.pdf')

    ze_p = P.zeros()
    po_P = P.poles()

    tau_0 = 1 / 90
    tau_1 = 1 / 5 
    tau_2 = 1 / 5  
    tau_3 = 1 / 5
    L_des = (s - ze_p[0]) / (tau_0 * (tau_1 * s + 1) * (tau_2 * s + 1) * (tau_3 * s + 1)) # can't cancel out CRHP zro in L

    # unstable part of P
    Pu = (s - ze_p[0])/((tau_1 * s + 1) * (tau_2 * s + 1))
    # stable part of P, minimum phase
    Pm = (tau_1 * s + 1) * (tau_2 * s + 1) * (s - ze_p[1]) * (s - ze_p[2]) / ((s - po_P[0]) * (s - po_P[1]) * (s - po_P[2]) * (s - po_P[3]))

    C1 = control.minreal(L_des/Pu)

    # Control design
    C = C1/Pm

    # Open-loop transfer function
    L = control.minreal(P * C)

    if investigate_ORHP:
        print(f'C = {C}\n')
        print(f'Zeros of C: {C.zeros()}')
        print(f'Poles of C: {C.poles()}')


    # Gang of four
    # T
    T = control.minreal(control.feedback(P * C))
    # S
    S = control.minreal(1 - T)
    # PS
    PS = control.minreal(P * S)
    # CS
    CS = control.minreal(C * S)

    # if investigate_ORHP:
    #     print(f'S = {S}')
    #     print(f'CS = {CS}')
    #     print(f'PS = {PS}')
    #     print(f'T = {T}\n')

    if show_graphs:
        # Individual S and T
        mag_abs, _, w = control.bode(S, w_shared, plot=False)
        mag_dB_S = 20 * np.log10(mag_abs)

        mag_abs, _, w = control.bode(T, w_shared, plot=False)
        mag_dB_T = 20 * np.log10(mag_abs)

        fig, ax = plt.subplots()
        # fig.set_size_inches(8.5, 11, forward=True)
        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
        # Magnitude plot (dB).
        ax.semilogx(w, mag_dB_S, '--', color='C3', label=r'$|S(j \omega)|$')
        ax.semilogx(w, mag_dB_T, '-.', color='C4', label=r'$|T(j \omega)|$')
        ax.legend(loc='best')
        fig.tight_layout()
        # fig.savefig('.pdf')

        # Bode magnitude plot of P(s), C(s), and L(s)
        mag_abs, _, w = control.bode(P, w_shared, plot=False)
        mag_dB_P = 20 * np.log10(mag_abs)

        mag_abs, _, w = control.bode(C, w_shared, plot=False)
        mag_dB_C = 20 * np.log10(mag_abs)

        mag_abs, _, w = control.bode(L, w_shared, plot=False)
        mag_dB_L = 20 * np.log10(mag_abs)

        fig, ax = plt.subplots()
        # fig.set_size_inches(8.5, 11, forward=True)
        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
        # Magnitude plot (dB).
        ax.semilogx(w, mag_dB_P, '-', color='C0', label=r'$|P(j \omega)|$')
        ax.semilogx(w, mag_dB_C, '--', color='C1', label=r'$|C(j \omega)|$')
        ax.semilogx(w, mag_dB_L, '-.', color='C2', label=r'$|L(j \omega)|$')
        ax.legend(loc='best')
        fig.tight_layout()
        # fig.savefig('.pdf')

        # Bode magnitude plot of W_2^-1(s),  W_1^-1(s)(s), and L(s)

        w_line = np.logspace(-3, 0, int(N_w/2))

        one_over_y_r = (1/gamma_r)* np.ones(int(N_w/2))
        mag_dB_one_over_y_r = 20 * np.log10(one_over_y_r)

        w2m1 = control.minreal(W2**-1)
        w1m1 = control.minreal(W1**-1)

        if investigate_ORHP:
            print(f'W2 = {W2}')
            print(f'W1 = {W1}')
            print(f'W2^-1 = {w2m1}')
            print(f'W1^-1 = {w1m1}')

        mag_abs, _, w = control.bode(w2m1, w_shared, plot=False)
        mag_dB_w2m1 = 20 * np.log10(mag_abs)

        mag_abs, _, w = control.bode(w1m1, w_shared, plot=False)
        mag_dB_w1m1 = 20 * np.log10(mag_abs)

        mag_abs, _, w = control.bode(L, w_shared, plot=False)
        mag_dB_L = 20 * np.log10(mag_abs)

        fig, ax = plt.subplots()
        # fig.set_size_inches(8.5, 11, forward=True)
        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
        # Magnitude plot (dB).
        ax.semilogx(w, mag_dB_w2m1, '-', color='C0', label=r'$|W_2^-1(j \omega)|$')
        ax.semilogx(w, mag_dB_w1m1, '--', color='C1', label=r'$|W_1^-1(j \omega)|$')
        ax.semilogx(w, mag_dB_L, '-.', color='C2', label=r'$|L(j \omega)|$')
        ax.semilogx(w_line, mag_dB_one_over_y_r, '.', color='C3', label=r'$1/\gamma_r$')
        ax.semilogx(np.array([np.real(ze_p[0])]), np.array([0]), 'x', color='C4', label=r'CRHP zero')
        ax.semilogx(w, mag_dB_C, '--', color='C5', label=r'$|C(j \omega)|$')

        ax.legend(loc='best')
        fig.tight_layout()
        # fig.savefig('.pdf')


        # print('Nyquist plot of L with model uncertainty')
        count, fig, ax = robust_nyq(L, W2, 1, 3, 250)
        fig.tight_layout()
        # fig.savefig('.pdf')

        # print('Nominal performance condition and robust stability condition')

        mag_abs_W1S, _, w = control.bode(W1*S, w_shared, plot=False)
        mag_dB_W1S = 20 * np.log10(mag_abs_W1S)

        mag_abs_W2T, _, w = control.bode(W2*T, w_shared, plot=False)
        mag_dB_W2T = 20 * np.log10(mag_abs_W2T)

        fig, ax = plt.subplots()
        # fig.set_size_inches(8.5, 11, forward=True)
        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
        # Magnitude plot (dB).
        ax.semilogx(w, mag_dB_W1S, '-', color='C0', label=r'$|W_1(j \omega) S(j \omega)|$')
        ax.semilogx(w, mag_dB_W2T, '-', color='C1', label=r'$|W_2(j \omega) T(j \omega)|$')
        ax.legend(loc='best')
        fig.tight_layout()
        # fig.savefig('separate_conditions.pdf')

        # print('Robust Performance condition')
        mag_abs_1p_L, _, w = control.bode(1 + L, w_shared, plot=False)
        mag_dB_1p_L = 20 * np.log10(mag_abs_1p_L)

        mag_abs_L_m_W2, _, w = control.bode(W2*L, w_shared, plot=False)
        mag_dB_W1_p_L_w2 = 20 * np.log10(mag_abs_W1 + mag_abs_L_m_W2)

        fig, ax = plt.subplots()
        # fig.set_size_inches(8.5, 11, forward=True)
        ax.set_xlabel(r'$\omega$ (rad/s)')
        ax.set_ylabel(r'$\gamma(\omega)$ (dB)')
        # Magnitude plot (dB).
        ax.semilogx(w, mag_dB_1p_L, '-', color='C0', label=r'$|1 + L(j \omega)|$')
        ax.semilogx(w, mag_dB_W1_p_L_w2, '-', color='C1', label=r'$|W_1(j \omega)| + |W_2(j \omega) L(j \omega)|$')
        ax.legend(loc='best')
        fig.tight_layout()
        # fig.savefig('Robust_Performance_condition.pdf')



    # Simulate closed-loop system.
    # Load profile to track
    data_ref = np.loadtxt('DATA/reference/ref.csv',
                            dtype=float,
                            delimiter=',',
                            skiprows=1,
                            usecols=(0, 1, ))

    # Extract time and reference data
    t = data_ref[:, 0]
    r = data_ref[:, 1]

    # For the purposes of testing the controller C(s), P_true(s) is considered the
    # ``true" plant.
    A_plant = np.array([[-4.64500000e+03, 1.06793820e+03, -5.56992280e+02,
            8.44638500e+01, 9.91716000e+01, -2.64000000e-01],
            [-1.00000000e+04, 4.47041308e-13, 5.59840915e-13,
            -4.87948758e-13, -8.19131992e-14, 1.27258391e-12],
            [ 0.00000000e+00, -1.00000000e+03, -1.47672753e-13,
            1.56997853e-13, -2.43528124e-13, -2.11672377e-13],
            [ 0.00000000e+00, 0.00000000e+00, -1.00000000e+03,
            -1.28128662e-13, 1.75012746e-13, 1.29344491e-13],
            [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            1.00000000e+02, 6.23467520e-14, -5.04317226e-14],
            [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, -1.00000000e+02, -1.60597507e-13]])
    B_plant = np.array([[0.01],
                        [0.],
                        [0.],
                        [0.],
                        [0.],
                        [0.]])
    C_plant = np.array([[1380.0, 275.36, 1044.8426, -296.20406, -213.4296, 139.568]])
    D_plant = np.array([[0]])
    P_true = control.ss2tf(A_plant, B_plant, C_plant, D_plant)  # Do NOT change this!

    # Compute closed-loop TFs using "true" plant.
    T_true = control.minreal(control.feedback(P_true * C))
    # Compute closed-loop tf from r to u using "true" plant
    CS_true = control.minreal(C /(1 + P_true * C))

    if investigate_ORHP:

        print(f'True P = {P_true}')
        print(f'Zeros of true tf: {P_true.zeros()}')
        print(f'Poles of true tf: {P_true.poles()}')
        print(f'Zeros of true T: {T_true.zeros()}')
        print(f'Poles of true T: {T_true.poles()}')
        print(f'Zeros of true CS: {CS_true.zeros()}')
        print(f'Poles of true CS: {CS_true.poles()}')

    if show_graphs:
        # Forced response of each system
        _, y = control.forced_response(T_true, t, r)
        e = r - y
        _, u = control.forced_response(CS_true, t, r)

        # Percent of full scale
        pFS = np.abs(e)/FS

        # Metrics
        e_mean = np.mean(e)
        e_std = np.std(e)
        e_max = np.max(np.abs(e))
        pFS_max = np.max(np.abs(pFS))
        u_max = np.max(np.abs(u))

        # Plot forced response
        fig, ax = plt.subplots(3, 1)
        fig.set_size_inches(8.5, 11, forward=True)
        ax[0].set_ylabel(r'Force (kN)')
        ax[1].set_ylabel(r'Voltage (V)')
        ax[2].set_ylabel(r'Perecent of Fullscale ($\%$)')
        # Plot data
        ax[0].plot(t, r, '--', label=r'$r(t)$ (kN)', color='C1')
        ax[0].plot(t, y, label=r'$y(t)$ (kN)', color='C0')
        ax[1].plot(t, u, label=r'$u(t)$ (V)', color='C2')
        ax[2].plot(t, pFS, label=r'$\%FS$', color='C3')
        for a in np.ravel(ax):
            a.set_xlabel(r'Time (s)')
            a.legend(loc='upper right')
        fig.tight_layout()
        # fig.savefig('.pdf')



        # Plot
        plt.show()
