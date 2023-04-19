import plant_nominal as fpnom
import numpy as np
import plant_util as util

from collections import Counter



from matplotlib import pyplot as plt

if __name__ == '__main__':

    a = np.array([1+2j, 
                  3+4j])

    print(fpnom._cmplxNbrsToPoints(a))

    b = np.array([[1., 2.], [3., 4.]])

    print(b)

    plant_list = util.load_plant_list_from_file('good_plants/v3/plants_trained_v3.json')
    plant_list = [p.delta_y_over_delta_u_c for p in plant_list]

    all_poles_cmplx = fpnom._getCmplxArrayFrmPlants(plant_list, fpnom._getPlantPoles)

    all_poles_points = fpnom._cmplxNbrsToPoints(all_poles_cmplx)

    print(all_poles_points)

    
    cluster_centers = fpnom._clusterPoints(all_poles_points, bandwidth=5)

    print(cluster_centers)
    # print(labels)
    # print(Counter(labels))

    fig, ax = plt.subplots()

    ax.plot(all_poles_points[:, 0], all_poles_points[:, 1], '+')
    ax.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'o', color='red')
    for a in np.ravel(ax):
        a.grid(visible=True)
    fig.tight_layout()


    all_zeros_cmplx = fpnom._getCmplxArrayFrmPlants(plant_list, fpnom._getPlantZeros)

    all_zeros_points = fpnom._cmplxNbrsToPoints(all_zeros_cmplx)

    print(all_zeros_points)

    
    cluster_centers = fpnom._clusterPoints(all_zeros_points, bandwidth=50, min_nbr_point=3)

    print(cluster_centers)
    # print(labels)
    # print(Counter(labels))

    fig, ax = plt.subplots()

    ax.plot(all_zeros_points[:, 0], all_zeros_points[:, 1], '+')
    ax.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'o', color='red')
    for a in np.ravel(ax):
        a.grid(visible=True)
    fig.tight_layout()

    plt.show()

    
