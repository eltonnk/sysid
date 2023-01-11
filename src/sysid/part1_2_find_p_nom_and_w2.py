import control
import numpy as np
from matplotlib import pyplot as plt
import plant_util as util
import unc_bound

import plant_nominal as nom


if __name__ == '__main__':
    
    MAIN_FILE_PATH = util.find_io_files_folder()
    METHOD = 3
    VERSION = 'v2'
    investigate_ORHP = True

    # Load trained plants
    plant_list = util.load_plant_list_from_file(MAIN_FILE_PATH / f'good_plants/{VERSION}/plants_trained_{VERSION}.json')
    plant_list = [p.delta_y_over_delta_u_c for p in plant_list]
    print(f'Number of identified plants: {len(plant_list)}')

    if METHOD == 1:
        P_nom = nom.plant_av_method_1(plant_list)
    elif METHOD == 2:
        
        P_nom = nom.plant_av_method_2(plant_list)
        if investigate_ORHP:
            print(f'Zeros of av tf: {P_nom.zeros()}')
            print(f'Poles of av tf: {P_nom.poles()}')
            print(f'Zeros of tf 0: {plant_list[0].zeros()}')
            print(f'Poles of tf 0: {plant_list[0].poles()}')
            print(f'Av tf: {P_nom}')
            print(f'tf 1: {plant_list[0]}')

    elif METHOD == 3:
        P_nom = nom.plant_av_method_3(plant_list)
        if investigate_ORHP:
            for i, p in enumerate(plant_list):
                for ze in p.zeros():
                    if np.real(ze) > 0:
                        print(f'Plant {i} has a zero with positive real part')
                for po in p.zeros():
                    if np.real(po) > 0:
                        print(f'Plant {i} has a pole with positive real part')

    # Method 4
    # P_nom = find_nom_plant_with_clust(plant_list, bandwidth_zeros=100, bandwidth_poles=5)

    print(f'Poles of reduced tf: {P_nom.poles()}')

    nom.plot_nom_vs_all(P_nom, plant_list)

    # Find the residuals
    list_R = unc_bound.residuals(P_nom, plant_list)
    # print(list_R[0])
    w_shared = unc_bound.generate_std_frequency_array()
    list_mag_abs = np.vstack([control.bode(residual, w_shared, plot=False)[0] for residual in list_R])
    list_mag_dB = 20 * np.log10(list_mag_abs)

    # Figure first simple bound
    show_simple_bound = False
    if show_simple_bound:
        s = control.tf('s')

        w_cross = 1.7e1
        y_low = 0.3
        y_high = 5.3
        W2_simple = (s+w_cross*y_low)/(s/y_high+w_cross)

        mag_abs_W2_simple = control.bode(W2_simple, w_shared, plot=False)[0]
        mag_dB_W2_simple = 20 * np.log10(mag_abs_W2_simple)

    
        fig, ax = plt.subplots(2, 1)
        ax[0].set_ylabel(r'$|R_k(j\omega)|$ (dB)')
        ax[1].set_ylabel(r'$||R_k(j\omega)|$ (absolute)')
        for mag_dB, mag_abs, index in zip(list_mag_dB, list_mag_abs, range(len(list_mag_dB))):
            ax[0].plot(w_shared, mag_dB, label=f'R{index}', color=f'C{index}')
            ax[1].plot(w_shared, mag_abs, label=f'R{index}', color=f'C{index}')
        for a in np.ravel(ax):
            a.set_xlabel(r'$\omega$ (rad/s)')
            a.set_xscale('log')
            a.grid(visible=True)
        ax[0].plot(w_shared, mag_dB_W2_simple, label=f'W2s', color=f'C{len(list_mag_dB)}')
        ax[1].plot(w_shared, mag_abs_W2_simple, label=f'W2s', color=f'C{len(list_mag_dB)}')   
        ax[0].legend(loc='lower right', ncol=3)
        ax[1].legend(loc='upper left', ncol=3)
        fig.tight_layout()

    # Find opitomal bound
    # Decide if want to compute and show optimal bound 
    # (Or just compute and show initial bound used as x0 for optimization)
    calc_optim = True
    

    # Initial bound coeffs
    kappa = 10**(-9/20)
    x0 = np.array([3e-1     , 0.9    , 1.1e2    , 1.1    , 8e-1     , 0.9    , 6.5e0    , 1.1    , kappa           ])
    # Optimization algo params
    lb = np.array([x0[0]/100, x0[1]-0.3, x0[2]/100, x0[3]-0.3, x0[4]/100, x0[5]-0.3, x0[6]/100, x0[7]-0.3, kappa-3**(-9/20)])
    ub = np.array([x0[0]*100, x0[1]+0.3, x0[2]*100, x0[3]+0.3, x0[4]*100, x0[5]+0.3, x0[6]*100, x0[7]+0.3, kappa+3**(-9/20)])
    max_iter = 100000

    if calc_optim:
        # Find max amplitude of residuals at all frequencies
        mag_dB_max, mag_abs_max = unc_bound.residual_max_mag(list_R, w_shared)
        # Compute optimal bound
        x_opt, f_opt, f_hist, x_hist = unc_bound.run_optimization(
            x0,
            lb,
            ub,
            max_iter,
            w_shared,
            mag_abs_max,
        )

    if calc_optim:
        # Build optimal bound tf using computed optimal coeffs
        W2_optim = unc_bound.extract_W2(x_opt)
        mag_abs_W2_optim = control.bode(W2_optim, w_shared, plot=False)[0]
        mag_dB_W2_optim = 20 * np.log10(mag_abs_W2_optim)
    else:
        # Build initial bound tf using inital coeffs to be used by 
        W2_init = unc_bound.extract_W2(x0)
        mag_abs_W2_init = control.bode(W2_init, w_shared, plot=False)[0]
        mag_dB_W2_init = 20 * np.log10(mag_abs_W2_init)

    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel(r'$|R_k(j\omega)|$ (dB)')
    ax[1].set_ylabel(r'$||R_k(j\omega)|$ (absolute)')
    for mag_dB, mag_abs, index in zip(list_mag_dB, list_mag_abs, range(len(list_mag_dB))):
        ax[0].plot(w_shared, mag_dB, label=f'R{index}', color=f'C{index}')
        ax[1].plot(w_shared, mag_abs, label=f'R{index}', color=f'C{index}')
    for a in np.ravel(ax):
        a.set_xlabel(r'$\omega$ (rad/s)')
        a.set_xscale('log')
        a.grid(visible=True)

    if calc_optim:
        ax[0].plot(w_shared, mag_dB_W2_optim, label=f'W2_optim', color=f'black')
        ax[1].plot(w_shared, mag_abs_W2_optim, label=f'W2_optim', color=f'black')
    else:
        ax[0].plot(w_shared, mag_dB_W2_init, label=f'W2_init', color=f'black')
        ax[1].plot(w_shared, mag_abs_W2_init, label=f'W2_init', color=f'black')

    ax[0].legend(loc='upper left', ncol=3)
    ax[1].legend(loc='upper left', ncol=3)
    fig.tight_layout()

    plt.show()

    p_nom_plant = util.Plant(P_nom,0,0,f'nominal_plant_{VERSION}')
    p_nom_plant.to_file(MAIN_FILE_PATH / f'good_plants/{VERSION}/P_nom_{VERSION}.json')

    w2_plant = util.Plant(W2_optim,0,0,f'optimal_boundary_W2_{VERSION}')
    w2_plant.to_file(MAIN_FILE_PATH / f'good_plants/{VERSION}/W2_{VERSION}.json')

    

    
