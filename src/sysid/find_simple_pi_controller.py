import plant_util as util
import pathlib
import control
import matplotlib.pyplot as plt


def PI_cntrlr_frm_coeffs(kp:float, ki:float):
    # derivation based on https://electronics.stackexchange.com/questions/9251/what-is-the-advantage-of-a-z-transform-derived-pid-implemenation
    z = control.tf('z')
    return ((_KP + delta_t*_KI)*z - 1) / (z - 1)

if __name__ == '__main__':

    MAIN_FILE_FOLDER = util.find_io_files_folder()

    VERSION = 'v1'

    REFERENCE_FILE_PATH = pathlib.Path(MAIN_FILE_FOLDER + 'DATA/reference/ref.csv')

    P_nom_plant = util.Plant.from_file(MAIN_FILE_FOLDER + f'good_plants/{VERSION}/P_nom_{VERSION}.json')
    P = P_nom_plant.delta_y_over_delta_u_c

    # frequency control loop
    f = 50000.0
    # sampling period
    delta_t = 1.0/f # 50 kHz

    print(P)
    P_d = control.c2d(P, delta_t)

    # Old controller
    # Old PI Values 
    _KP = 1.5
    _KI = 40.0

    old_controller = PI_cntrlr_frm_coeffs(_KP, _KI)

    old_closed_loop = control.feedback(old_controller*P_d)

    t, y = control.step_response(old_closed_loop)

    # New controller

    # Old PI Values 
    _KP = 1.5
    _KI = 40.0





    plt.plot(t, y, color='C1', label='Old controller')
    plt.legend()


    plt.show()