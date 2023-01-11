import control
import numpy as np
from typing import List, Callable
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sklearn.cluster import MeanShift

def _getNumCoeffList(p: control.TransferFunction) -> List[float]:
    return p.num[0][0]

def _getDenCoeffList(p: control.TransferFunction) -> List[float]:
    return p.den[0][0]

def _getAverageCoeffArray(
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

def _getPlantZeros(p: control.TransferFunction) -> List[complex]:
    return p.zeros()

def _getPlantPoles(p: control.TransferFunction) -> List[complex]:
    return p.poles()

def _getCmplxArrayFrmPlants(
    plant_list: List[control.TransferFunction], 
    fromWhereFunc: Callable[[control.TransferFunction], List[float]]
) -> np.ndarray:
    all_cmplx = []
    for p in plant_list:
        all_cmplx.extend(fromWhereFunc(p))

    return(np.array(all_cmplx))

def _cmplxNbrsToPoints(all_cmplx: np.ndarray) -> np.ndarray:

    all_real = np.reshape(np.real(all_cmplx),(all_cmplx.shape[0],1))
    all_imag = np.reshape(np.imag(all_cmplx),(all_cmplx.shape[0],1))

    all_points = np.concatenate([all_real, all_imag], axis=1)

    return all_points

def _clusterPoints(all_points: np.ndarray, bandwidth: float, min_nbr_point:float=1) -> np.ndarray:
    clustering  = MeanShift(bandwidth=bandwidth,min_bin_freq=min_nbr_point,bin_seeding=True,cluster_all=False).fit(all_points)

    labels = clustering.labels_ #TODO: eliminate some poles/zeros if not enough points
    return clustering.cluster_centers_

def plant_av_method_1(plant_list: List[control.TransferFunction]) -> control.TransferFunction:
    # Method 1
    # Add all plants and divide by number of plants, then simplify order to 4th order
    P_nom = plant_list[0]
    for p in plant_list[1:]:
        P_nom = P_nom + p
    P_nom = P_nom / len(plant_list)
    # print(f'Poles of unreduced tf: {P_nom.poles()}')
    # Reduce plant order to make it around same order as other plants
    P_nom = control.tf2ss(P_nom)
    P_nom = control.balred(P_nom,orders=4)
    P_nom = control.ss2tf(P_nom)

    return P_nom

def plant_av_method_2(plant_list: List[control.TransferFunction]) -> control.TransferFunction:
    # Method 2
    # Find tf from average of coefficients
    num_P_nom = _getAverageCoeffArray(plant_list, _getNumCoeffList)
    den_P_nom = _getAverageCoeffArray(plant_list, _getDenCoeffList)
    P_nom_av = control.tf(num_P_nom, den_P_nom)
    
    return P_nom_av

def plant_av_method_3(plant_list: List[control.TransferFunction]) -> control.TransferFunction:
    # Combine method 1 and 2
    P_nom = plant_av_method_2(plant_list)
        
    for i, p in enumerate(plant_list):
        P_nom = P_nom + p
    P_nom = P_nom / len(plant_list)
    # Reduce plant order to make it around same order as other plants
    P_nom = control.tf2ss(P_nom)
    P_nom = control.balred(P_nom,orders=4)
    P_nom = control.ss2tf(P_nom)

    return P_nom

def find_nom_plant_with_clust(
    plant_list: List[control.TransferFunction], 
    bandwidth_zeros: float, 
    bandwidth_poles: float
) -> control.TransferFunction:
    # Find nominal plant with clustering
    # DOES NOT WORK FOR NOW : works well with poles but not zeros
    all_zeros_cmplx = _getCmplxArrayFrmPlants(plant_list, _getPlantZeros)
    all_zeros_points = _cmplxNbrsToPoints(all_zeros_cmplx)
    zeros_cluster_centers = _clusterPoints(all_zeros_points, bandwidth=bandwidth_zeros, min_nbr_point=3)
    
    all_poles_cmplx = _getCmplxArrayFrmPlants(plant_list, _getPlantPoles)
    all_poles_points = _cmplxNbrsToPoints(all_poles_cmplx)
    poles_cluster_centers = _clusterPoints(all_poles_points, bandwidth=bandwidth_poles, min_nbr_point=3)

    num = np.poly(np.ravel(zeros_cluster_centers))
    den = np.poly(np.ravel(poles_cluster_centers))
    return control.tf(num, den)

def generate_std_frequency_array() -> np.ndarray:
    w_shared = np.arange(0.01, 1000.0, 0.01)
    return w_shared

def setup_magnitude_plots() ->  tuple[Figure, list[Axes]]:
    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel(r'$|R_k(j\omega)|$ (dB)')
    ax[1].set_ylabel(r'$||R_k(j\omega)|$ (absolute)')

    return fig, ax

def add_single_plant_to_mag_plot(
    w_shared:np.ndarray,
    ax: list[Axes],
    plant: control.TransferFunction,
    custom_plant_name: str,
    custom_color: str
):
    mag_abs_nom = control.bode(plant, w_shared, plot=False)[0]
    mag_dB_nom = 20 * np.log10(mag_abs_nom)
    ax[0].plot(w_shared, mag_dB_nom, label=custom_plant_name, color=custom_color)
    ax[1].plot(w_shared, mag_abs_nom, label=custom_plant_name, color=custom_color)


def add_list_plant_to_mag_plot(
    w_shared:np.ndarray,
    ax: list[Axes], 
    plant_list: list[control.TransferFunction],
    custom_plant_names: list[str] = []
):
    list_mag_abs = np.vstack([control.bode(plant, w_shared, plot=False)[0] for plant in plant_list])
    list_mag_dB = 20 * np.log10(list_mag_abs)
    for mag_dB, mag_abs, index in zip(list_mag_dB, list_mag_abs, range(len(list_mag_dB))):
        label = custom_plant_names[index] if custom_plant_names else f'P{index}'
        ax[0].plot(w_shared, mag_dB, label=label, color=f'C{index}')
        ax[1].plot(w_shared, mag_abs, label=label, color=f'C{index}')

def finish_magnitude_plots(fig:Figure, ax:Axes):
    for a in np.ravel(ax):
        a.set_xlabel(r'$\omega$ (rad/s)')
        a.set_xscale('log')
        a.grid(visible=True)
    ax[0].legend(loc='lower left', ncol=3)
    ax[1].legend(loc='upper right', ncol=3)
    fig.tight_layout()

def plot_nom_vs_all(P_nom: control.TransferFunction, plant_list: list[control.TransferFunction]):
    # Plot nominal plant vs all other plants
    fig, ax, w_shared = setup_magnitude_plots()
    add_list_plant_to_mag_plot(w_shared, ax, plant_list)
    add_single_plant_to_mag_plot(w_shared, ax, P_nom, f'P_nom', f'black')
    finish_magnitude_plots(fig, ax)