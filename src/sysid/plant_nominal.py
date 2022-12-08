import control
import numpy as np
from typing import List, Callable

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
    num_P_nom = getAverageCoeffArray(plant_list, getNumCoeffList)
    den_P_nom = getAverageCoeffArray(plant_list, getDenCoeffList)
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