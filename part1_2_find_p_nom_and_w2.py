import control
import numpy as np
from matplotlib import pyplot as plt
import plant_util as util
from typing import List, Callable
import unc_bound

from sklearn.cluster import MeanShift

def getNumCoeffList(p: control.TransferFunction) -> List[float]:
    return p.num[0][0]

def getDenCoeffList(p: control.TransferFunction) -> List[float]:
    return p.den[0][0]

def getAverageCoeffArray(
    plant_list: List[control.TransferFunction], 
    fromWhereFunc: Callable[[control.TransferFunction], List[float]]
) -> np.ndarray:
    
    # find num or denom order for all plants
    list_size_arr = []
    for p in plant_list:
        list_size_arr.append(fromWhereFunc(p).size)

    list_av_coef = []
    # we find average of all coefficients. We do this for x coefficients, where x
    # is the highest num or denom for all plants + 1
    for index in range(max(list_size_arr)):
        try: 
            list_av_coef.append(np.mean([fromWhereFunc(p)[index] for p in plant_list]))
        except:
            list_av_coef.insert(0,0)
    return np.array(list_av_coef)

def getPlantZeros(p: control.TransferFunction) -> List[complex]:
    return p.zeros()

def getPlantPoles(p: control.TransferFunction) -> List[complex]:
    return p.poles()

def getCmplxArrayFrmPlants(
    plant_list: List[control.TransferFunction], 
    fromWhereFunc: Callable[[control.TransferFunction], List[float]]
) -> np.ndarray:
    all_cmplx = []
    for p in plant_list:
        all_cmplx.extend(fromWhereFunc(p))

    return(np.array(all_cmplx))

def cmplxNbrsToPoints(all_cmplx: np.ndarray) -> np.ndarray:

    all_real = np.reshape(np.real(all_cmplx),(all_cmplx.shape[0],1))
    all_imag = np.reshape(np.imag(all_cmplx),(all_cmplx.shape[0],1))

    all_points = np.concatenate([all_real, all_imag], axis=1)

    return all_points

def clusterPoints(all_points: np.ndarray, bandwidth: float, min_nbr_point:float=1) -> np.ndarray:
    clustering  = MeanShift(bandwidth=bandwidth,min_bin_freq=min_nbr_point,bin_seeding=True,cluster_all=False).fit(all_points)

    labels = clustering.labels_ #TODO: eliminate some poles/zeros if not enough points
    return clustering.cluster_centers_

def find_nom_plant_with_clust(plant_list: List[control.TransferFunction], bandwidth_zeros: float, bandwidth_poles: float) -> control.TransferFunction:
    all_zeros_cmplx = getCmplxArrayFrmPlants(plant_list, getPlantZeros)
    all_zeros_points = cmplxNbrsToPoints(all_zeros_cmplx)
    zeros_cluster_centers = clusterPoints(all_zeros_points, bandwidth=bandwidth_zeros, min_nbr_point=3)
    
    all_poles_cmplx = getCmplxArrayFrmPlants(plant_list, getPlantPoles)
    all_poles_points = cmplxNbrsToPoints(all_poles_cmplx)
    poles_cluster_centers = clusterPoints(all_poles_points, bandwidth=bandwidth_poles, min_nbr_point=3)

    num = np.poly(np.ravel(zeros_cluster_centers))
    den = np.poly(np.ravel(poles_cluster_centers))
    return control.tf(num, den)

if __name__ == '__main__':
    
    MAIN_FILE_FOLDER = util.find_io_files_folder()
    METHOD = 2
    VERSION = 'v1'
    investigate_ORHP = True

    # Load trained plants
    plant_list = util.load_plant_list_from_file(MAIN_FILE_FOLDER + f'good_plants/{VERSION}/plants_trained_{VERSION}.json')
    plant_list = [p.delta_y_over_delta_u_c for p in plant_list]
    print(f'Number of identified plants: {len(plant_list)}')

    if METHOD == 1:
        # Method 1
        # Add all plants and divide by number of plants, then simplify order to 4th order
        P_nom = plant_list[0]
        for p in plant_list[1:]:
            P_nom = P_nom + p
        P_nom = P_nom / len(plant_list)
        print(f'Poles of unreduced tf: {P_nom.poles()}')
        # Reduce plant order to make it around same order as other plants
        P_nom = control.tf2ss(P_nom)
        P_nom = control.balred(P_nom,orders=4)
        P_nom = control.ss2tf(P_nom)
    elif METHOD == 2:
        # Method 2
        # Find tf from average of coefficients
        num_P_nom = getAverageCoeffArray(plant_list, getNumCoeffList)
        den_P_nom = getAverageCoeffArray(plant_list, getDenCoeffList)
        P_nom_av = control.tf(num_P_nom, den_P_nom)
        if investigate_ORHP:
            print(f'Zeros of av tf: {P_nom_av.zeros()}')
            print(f'Poles of av tf: {P_nom_av.poles()}')
            print(f'Zeros of tf 0: {plant_list[0].zeros()}')
            print(f'Poles of tf 0: {plant_list[0].poles()}')
            print(f'Av tf: {P_nom_av}')
            print(f'tf 1: {plant_list[0]}')

        P_nom = P_nom_av

    elif METHOD == 3:
        # Combine method 1 and 2
        
        for i, p in enumerate(plant_list):
            P_nom = P_nom + p
            if investigate_ORHP:
                for ze in p.zeros():
                    if np.real(ze) > 0:
                        print(f'Plant {i} has a zero with positive real part')
                for po in p.zeros():
                    if np.real(po) > 0:
                        print(f'Plant {i} has a pole with positive real part')
        P_nom = P_nom / len(plant_list)
        # Reduce plant order to make it around same order as other plants
        P_nom = control.tf2ss(P_nom)
        P_nom = control.balred(P_nom,orders=4)
        P_nom = control.ss2tf(P_nom)


    # Method 4
    # P_nom = find_nom_plant_with_clust(plant_list, bandwidth_zeros=100, bandwidth_poles=5)

    print(f'Poles of reduced tf: {P_nom.poles()}')

    # Plot nominal plant vs all other plants
    w_shared = np.arange(0.01, 1000.0, 0.01)
    list_mag_abs = np.vstack([control.bode(plant, w_shared, plot=False)[0] for plant in plant_list])
    list_mag_dB = 20 * np.log10(list_mag_abs)
    mag_abs_nom = control.bode(P_nom, w_shared, plot=False)[0]
    mag_dB_nom = 20 * np.log10(mag_abs_nom)

    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel(r'$|R_k(j\omega)|$ (dB)')
    ax[1].set_ylabel(r'$||R_k(j\omega)|$ (absolute)')
    for mag_dB, mag_abs, index in zip(list_mag_dB, list_mag_abs, range(len(list_mag_dB))):
        ax[0].plot(w_shared, mag_dB, label=f'P{index}', color=f'C{index}')
        ax[1].plot(w_shared, mag_abs, label=f'P{index}', color=f'C{index}')
    ax[0].plot(w_shared, mag_dB_nom, label=f'P_nom', color=f'black')
    ax[1].plot(w_shared, mag_abs_nom, label=f'P_nom', color=f'black')
    for a in np.ravel(ax):
        a.set_xlabel(r'$\omega$ (rad/s)')
        a.set_xscale('log')
        a.grid(visible=True)
    ax[0].legend(loc='lower left', ncol=3)
    ax[1].legend(loc='upper right', ncol=3)
    fig.tight_layout()

    # Find the residuals
    list_R = unc_bound.residuals(P_nom, plant_list)
    # print(list_R[0])
    w_shared = np.arange(0.01, 1000.0, 0.01)
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
    x0 = np.array([3e-1, 0.9, 7.5e1, 1.1, 8e-1, 0.9, 7.2e0, 1.1, kappa])
    # Optimization algo params
    lb = 1e-2
    ub = 1e3
    max_iter = 100000

    if calc_optim:
        # Find max amplitude of residuals at all frequencies
        mag_dB_max, mag_abs_max = unc_bound.residual_max_mag(list_R, w_shared)
        # Compute optimal bound
        x_opt, f_opt, objhist, xlast = unc_bound.run_optimization(
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
    p_nom_plant.to_file(MAIN_FILE_FOLDER + f'good_plants/{VERSION}/P_nom_{VERSION}.json')

    w2_plant = util.Plant(W2_optim,0,0,f'optimal_boundary_W2_{VERSION}')
    w2_plant.to_file(MAIN_FILE_FOLDER + f'good_plants/{VERSION}/W2_{VERSION}.json')

    

    
