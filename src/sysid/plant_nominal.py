import control
import numpy as np
from typing import List, Callable
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation

from sklearn.cluster import MeanShift

def _getNumCoeffList(p: control.TransferFunction) -> List[float]:
    return p.num[0][0]

def _getDenCoeffList(p: control.TransferFunction) -> List[float]:
    return p.den[0][0]

def _getAverageCoeffArray(
    plant_list: List[control.TransferFunction], 
    fromWhereFunc: Callable[[control.TransferFunction], List[float]],
    zero_out_phase: bool = False
) -> np.ndarray:
    """Finds a list of coefficients for the numerator o denominator of 
    the nominal plant in a list of plants

    Parameters
    ----------
    plant_list : List[control.TransferFunction]
        List of identified plants, from which we want to find a average
        numerator or denominator coefficient
    fromWhereFunc : Callable[[control.TransferFunction], List[float]]
        Should either be  _getNumCoeffList to average out numerator coefficients 
        or should be_getDenCoeffList to average out denominator coefficients.
    zero_out_phase : bool, optional
        Should only be used when fromWhereFunc=_getNumCoeffList and to mitigate
        phase issues due to delay introduced by hardware in input/output data 
        used during plant identification process. Since we don't consider phase 
        when computing uncertainty boundary, removing 180 degrees from phase 
        makes no difference when computing gain bode plots to compute W2, but 
        make a big difference when finding nominal plant by averaging out tf 
        coefficients when using  plant_av_method_2. Phase issues can be detected 
        when plants have similar numerators, but some have all positive 
        coefficients, and some have all negative coefficients. By default, 
        False.

    Returns
    -------
    np.ndarray
        List of coefficients of the average numerator or denominator coefficient.
    """
    # find num or denom order for all plants
    coeff_arrays = []
    list_size_arr = []
    for p in plant_list:
        coeff_array = fromWhereFunc(p)
        if zero_out_phase and all(coeff_array < 0):
            coeff_array = -1*coeff_array
        coeff_arrays.append(coeff_array)
        list_size_arr.append(coeff_array.size)

    list_av_coef = []
    # we find average of all coefficients. We do this for x coefficients, where x
    # is the highest num or denom for all plants + 1

    for index in range(max(list_size_arr)):
        try: 
            list_av_coef.append(np.mean([coeff_array[index] for coeff_array in coeff_arrays]))
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

def plot_poles(plant_list: List[control.TransferFunction], custom_plant_names: list[str] = [], plot_cluster_center: bool = False):
    all_poles_cmplx = _getCmplxArrayFrmPlants(plant_list, _getPlantPoles)
    pnbr = int(len(all_poles_cmplx) / len(plant_list))

    all_poles_points = _cmplxNbrsToPoints(all_poles_cmplx)
    print("Poles of all plants: ")
    print(all_poles_points)

    if plot_cluster_center:
        cluster_centers = _clusterPoints(all_poles_points, bandwidth=5)
        print("Center point of pole clusters:")
        print(cluster_centers)

    fig, ax = plt.subplots()
    for index, _ in enumerate(plant_list):
        label = custom_plant_names[index] if custom_plant_names else f'P{index}'
        ax.plot(all_poles_points[index*pnbr:(index+1)*pnbr, 0], all_poles_points[index*pnbr:(index+1)*pnbr, 1], '+', label=label, color=f'C{index}')
    if plot_cluster_center:
        ax.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'o', color='black')
    for a in np.ravel(ax):
        a.grid(visible=True)
    ax.legend(loc='upper left', ncol=3)
    fig.tight_layout()
    
def plot_zeros(plant_list: List[control.TransferFunction], custom_plant_names: list[str] = [], plot_cluster_center: bool = False):
    all_zeros_cmplx = _getCmplxArrayFrmPlants(plant_list, _getPlantZeros)
    znbr = int(len(all_zeros_cmplx) / len(plant_list))

    all_zeros_points = _cmplxNbrsToPoints(all_zeros_cmplx)
    print("Zeros of all plants: ")
    print(all_zeros_points)

    if plot_cluster_center:
        print("Center point of zero clusters:")
        cluster_centers = _clusterPoints(all_zeros_points, bandwidth=50, min_nbr_point=3)
        print(cluster_centers)

    fig, ax = plt.subplots()

    for index, _ in enumerate(plant_list):
        label = custom_plant_names[index] if custom_plant_names else f'P{index}'
        ax.plot(all_zeros_points[index*znbr:(index+1)*znbr, 0], all_zeros_points[index*znbr:(index+1)*znbr, 1], '+', label=label, color=f'C{index}')
    if plot_cluster_center:
        ax.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'o', color='red')
    for a in np.ravel(ax):
        a.grid(visible=True)
    ax.legend(loc='upper left', ncol=3)
    fig.tight_layout()


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

def plant_av_method_2(plant_list: List[control.TransferFunction], zero_out_phase: bool = False) -> control.TransferFunction:
    """Takes a list of plants and finds the nominal plant, by averaging plant coefficients.

    Works best when all plants have the same numerator and denominator coefficients.

    Parameters
    ----------
    plant_list : List[control.TransferFunction]
        List of identified plants, from which we want to find an average plant
    zero_out_phase : bool, optional
        Should only be used to mitigate phase issues due to delay introduced by 
        hardware in input/output data used during plant identification process. 
        Since we don't consider phase when computing uncertainty boundary, 
        removing 180 degrees from phase makes no difference when computing gain
        bode plots to compute W2, but make a big difference when finding 
        nominal plant by averaging out tf coefficients when using 
        plant_av_method_2. Phase issues can be detected when plants have similar
        numerators, but some have all positive coefficients, and some have all 
        negative coefficients. By default, False.

    Returns
    -------
    control.TransferFunction
        The nominal plant
    """
    # Method 2
    # Find tf from average of coefficients
    num_P_nom = _getAverageCoeffArray(plant_list, _getNumCoeffList, zero_out_phase=zero_out_phase)
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

def generate_std_frequency_array(start_freq = 0.01, stop_frequency = 1000.0, step=0.01) -> np.ndarray:
    w_shared = np.arange(start_freq, stop_frequency, step)
    return w_shared

def setup_magnitude_plots() ->  tuple[Figure, list[Axes]]:
    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel(r'$|P(j\omega)|$ (dB)')
    ax[1].set_ylabel(r'$|P(j\omega)|$ (absolute)')

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
    custom_plant_names: str | list[str] = None,
    custom_plant_colors: str | list[str] = None,
) -> tuple[list[Line2D], list[Line2D]]:
    
    names_is_list = isinstance(custom_plant_names, list)
    names_is_str = isinstance(custom_plant_names, str)
    colors_is_list = isinstance(custom_plant_colors, list)
    colors_is_str = isinstance(custom_plant_colors, str)

    list_mag_abs = np.vstack([control.bode(plant, w_shared, plot=False)[0] for plant in plant_list])
    list_mag_dB = 20 * np.log10(list_mag_abs)
    lines_dB = []
    lines_abs = []
    for mag_dB, mag_abs, index in zip(list_mag_dB, list_mag_abs, range(len(list_mag_dB))):
        if names_is_list:
            label = custom_plant_names[index]
        elif names_is_str:
            label = custom_plant_names
        else:
            label = f'P{index}'

        if colors_is_list:
            color = custom_plant_colors[index]
        elif colors_is_str:
            color = custom_plant_colors
        else:
            color = f'C{index}'

        if names_is_str and index != 0:
            ax[0].plot(w_shared, mag_dB, color=color)
            ax[1].plot(w_shared, mag_abs, color=color)
        else:
            ax[0].plot(w_shared, mag_dB, label=label, color=color)
            ax[1].plot(w_shared, mag_abs, label=label, color=color)
    
    return lines_dB, lines_abs

def finish_magnitude_plots(fig:Figure, ax:list[Axes], line_tuple: tuple[list[Line2D], list[Line2D]] = None):
    for a in np.ravel(ax):
        a.set_xlabel(r'$\omega$ (rad/s)')
        a.set_xscale('log')
        a.grid(visible=True)
    if not line_tuple:
        ax[0].legend(loc='lower left', ncol=3)
        ax[1].legend(loc='upper right', ncol=3)
        fig.tight_layout()
        return 

    (lines_dB, lines_abs) = line_tuple

    annot_dB = ax[0].annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot_dB.set_visible(False)

    annot_abs = ax[0].annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot_abs.set_visible(False)

    def update_annot(annot: Annotation, line: Line2D, idx:int):
        posx, posy = [line.get_xdata()[idx], line.get_ydata()[idx]]
        annot.xy = (posx, posy)
        text = f"{line.get_label()}"
        annot.set_text(text)
        # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        for a, annot, lines in zip(ax, [annot_dB, annot_abs], [lines_dB, lines_abs]):
            vis = annot.get_visible()
            if event.inaxes == a:
                for line in lines:
                    cont, ind = line.contains(event)
                    if cont:
                        update_annot(annot, line, ind["ind"][0])
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    fig.tight_layout()

def plot_nom_vs_all(P_nom: control.TransferFunction, plant_list: list[control.TransferFunction]):
    # Plot nominal plant vs all other plants
    w_shared = generate_std_frequency_array()
    fig, ax = setup_magnitude_plots()
    add_list_plant_to_mag_plot(w_shared, ax, plant_list)
    add_single_plant_to_mag_plot(w_shared, ax, P_nom, f'P_nom', f'black')
    finish_magnitude_plots(fig, ax)