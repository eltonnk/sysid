import pytest
import control
import numpy as np
from sysid import plant_util as util
from sysid import d2c

def _tf_equal(tf_original: control.TransferFunction, tf_recreated: control.TransferFunction):
    # See if they look similar
    is_equal = tf_original.__repr__() == tf_recreated.__repr__()
    if is_equal: # If they do, we have the same tf
        return is_equal
    # If they aren't identical, look to see if the relative difference in coeffs is less than a percent
    is_equal = True 
    for n_o, n_r, d_o, d_r in zip (tf_original.num[0][0], tf_recreated.num[0][0], tf_original.den[0][0], tf_recreated.den[0][0]):
        is_equal = True if abs(n_o - n_r) / n_o <= 0.01 else False
        is_equal = True if abs(d_o - d_r) / d_o <= 0.01 else False
    return is_equal

def _basic_tf():
    s: control.TransferFunction = control.tf('s')

    TF_OPTION = 'b'

    # System used as an example is a brushed DC motor
    if TF_OPTION == 'a':
        K = 1.25
        L = 0.02
        R = 0.5
        J = 0.1
        b = 1
        H_elec = K / (L*s + R)
        H_mech = 1 / (J*s + b)
        K_emf =  1.25

        input_to_torque_tf = control.feedback(H_elec, H_mech*K_emf)

        a_2 = 1/(input_to_torque_tf.den[0][0][0])

        input_to_torque_tf = control.tf(a_2 * input_to_torque_tf.num[0][0], a_2 * input_to_torque_tf.den[0][0])
        return input_to_torque_tf

    elif TF_OPTION == 'b':
        s = control.tf('s')
        omega = 10  # rad/s
        zeta = 0.25
        tau = 1 / 5
        Pa = omega ** 2 * (tau * s + 1) / (s ** 2 + 2 * zeta * omega * s + omega ** 2)

        return Pa

@pytest.fixture
def basic_tf():
    return _basic_tf()

def test_tf_encoder_decoder(basic_tf):
    str_test = util.tf_encoder(basic_tf)
    basic_tf_recreated = util.tf_decoder(str_test)

    assert(_tf_equal(basic_tf, basic_tf_recreated)) # best we can do since __eq__() is not implented in control


def test_cd2_d2c(basic_tf):
    dt = 0.0001
    basic_tf_discrete = control.c2d(basic_tf, Ts=dt)
    basic_tf_recreated = d2c.d2c(basic_tf_discrete)

    assert(_tf_equal(basic_tf, basic_tf_recreated))



def test_organized_sensor_data():
    #           u0 u1 u2 u3 u4
    u=np.array([0, 1, 2, 3, 4])
    #           y0 y1 y2 y3 y4
    y=np.array([5, 6, 7, 8, 9])
    sensor_data_test = util.SensorData.from_timeseries(
        t=np.array([0, 0.02, 0.04, 0.06, 0.08]), 
        r=np.array([0, 0, 0, 0, 0]),
        u=u,
        y=y,
    )
    design_params = util.PlantDesignParams(
        num_order=1,
        denum_order=2,
        better_cond_method=None,
        regularization=0.0,
        sensor_data_column_names=None,
    )
    
    organized_data = util.PlantOrganizedSensorData(sensor_data_test, design_params)

    assert((organized_data.b == np.array([
        [y[4]],
        [y[3]],
        [y[2]],
    ])).all())

    assert((organized_data.A == np.array([
        [-y[3], -y[2], u[3], u[2]],
        [-y[2], -y[1], u[2], u[1]],
        [-y[1], -y[0], u[1], u[0]],
    ])).all())

def _lchirp(N, tmin=0, tmax=1, fmin=0, fmax=None):
    fmax = fmax if fmax is not None else N / 2
    t = np.linspace(tmin, tmax, N, endpoint=True)

    a = (fmin - fmax) / (tmin - tmax)
    b = (fmin*tmax - fmax*tmin) / (tmax - tmin)

    phi = (a/2)*(t**2 - tmin**2) + b*(t - tmin)
    phi *= (2*np.pi)
    return phi

def lchirp(N, tmin=0, tmax=1, fmin=0, fmax=None, zero_phase_tmin=True, cos=True):
    phi = _lchirp(N, tmin, tmax, fmin, fmax)
    if zero_phase_tmin:
        phi *= ( (phi[-1] - phi[-1] % (2*np.pi)) / phi[-1] )
    else:
        phi -= (phi[-1] % (2*np.pi))
    fn = np.cos if cos else np.sin
    return fn(phi)


def test_correct_plant_deduced(basic_tf):
    DEBUGGING = True

    ID_SEQ_OPTION = 'c'

    if ID_SEQ_OPTION == 'a':
        # Generate input signal (PRBS)
        actuator_max_torque = 0.5
        id_seq_lenght=20.0 # seconds
        
        ctrl_period = 200 # microseconds
        sampling_period = ctrl_period * 1e-6 # seconds

        one_bit_period= 10 *  sampling_period

        N = int((1/sampling_period)*id_seq_lenght)
        one_bit_N = int((1/sampling_period)*one_bit_period)

        n = 0
        output_mem = 0.0

        start = 2
        a = start
        newbit = 0

        prbs = np.zeros((N,))
        t_step = np.zeros((N,))
        t_now = 0
        for n in range(N):
            if n % one_bit_N == 0 :
                newbit = (((a >> 6) ^ (a >> 5)) & 1)
                a = ((a << 1) | newbit) & 0x7f
                output_mem = ((2.0 * newbit) - 1.0) * actuator_max_torque
            prbs[n] = output_mem
            t_step[n] = t_now
            t_now += sampling_period


        dt = sampling_period
        x0 = np.array([0, 0]) # 2 states since 2nd order denominator for basic_tf
        basic_tf_discrete: control.TransferFunction = control.c2d(basic_tf, Ts=dt)
        result_step: control.TimeResponseData = control.forced_response(basic_tf_discrete, T=t_step,U=prbs, X0=x0)

    elif ID_SEQ_OPTION == 'b':
        dt = 0.05
        t_start = 0
        t_end = 5
        ta = np.arange(t_start, t_end, dt)
        N_t = ta.size
        jump_data = np.array([0.9, -0.1, 0.2, -0.8, 0.6, 0.1, -0.5, -0.25, 0.2, -0.1])
        N_jumps = jump_data.size
        ua = [0]
        for i in range(N_jumps):
            for j in range(int(N_t / N_jumps)):
                ua.append(jump_data[i])

        ua = ua[1:]  # remove 0th element
        ua = np.array(ua)

        basic_tf_discrete: control.TransferFunction = control.c2d(basic_tf, Ts=dt)
        x0 = np.array([0, 0]) # 2 states since 2nd order denominator for basic_tf
        result_step: control.TimeResponseData  = control.forced_response(basic_tf_discrete, T=ta, U=ua, X0=x0)
    
    elif ID_SEQ_OPTION == 'c':
        #chirp
        actuator_max_torque = 0.5
        ctrl_period = 200 # microseconds
        sampling_period = ctrl_period * 1e-6 # seconds
        data_collection_timespan = 25 # seconds
        delta_t = sampling_period # seconds ( corresponds to approximate transmission frequency)


        # This time_signal is fake in the fact that we don't know 
        # how fast the Inverse3 will process the commands

        time_signal = np.arange(start=0, step=delta_t, stop=data_collection_timespan)
        # time_signal = np.linspace(
        #     0, 
        #     data_collection_timespan, 
        #     int(data_collection_timespan/delta_t)
        # )
        nbr_chirps = 6
        v_at_zero_prct = 0.3
        zero2chirp_split = int(len(time_signal)/(nbr_chirps*20))
        single_sg_split = int(len(time_signal)/(nbr_chirps))


        vzero_sg = np.zeros(zero2chirp_split)
        single_chirp_timespan = data_collection_timespan*(1/nbr_chirps)*(1- v_at_zero_prct)
        single_chirp_nbr_points = single_sg_split - zero2chirp_split

        start_freq = 0.01
        end_freq = 1000

        # from https://dsp.stackexchange.com/questions/75282/end-of-chirp-in-phase-0

        vchirp_sg = actuator_max_torque*lchirp(
            N=single_chirp_nbr_points, 
            tmin=0, 
            tmax=single_chirp_timespan, 
            fmin=start_freq, 
            fmax=end_freq, 
            zero_phase_tmin=True, 
            cos=False
        )

        v_sg_blocks = [vzero_sg, vchirp_sg] * nbr_chirps


        voltage_signal = np.block(v_sg_blocks)

        diff_len =  abs(len(time_signal) - len(voltage_signal))
        if diff_len:
            voltage_signal = np.concatenate((voltage_signal, np.array([0] * diff_len)))

        basic_tf_discrete: control.TransferFunction = control.c2d(basic_tf, Ts=delta_t)
        x0 = np.array([0, 0]) # 2 states since 2nd order denominator for basic_tf
        result_step: control.TimeResponseData = control.forced_response(basic_tf_discrete, T=time_signal, U=voltage_signal, X0=x0)

    sensor_data_test = util.SensorData.from_timeseries(
        t=result_step.time, 
        r=result_step.inputs, 
        u=result_step.inputs, 
        y=result_step.outputs
    )

    if DEBUGGING:
        sensor_data_test.plot()
    
    
    design_outcome = util.PlantDesignOutcome('test_plant')

    
    num_order_discrete = len(basic_tf_discrete.num[0][0])-1
    denum_order_discrete = len(basic_tf_discrete.den[0][0])-1
    #reg_arr = util.build_regularization_array(-6, -3, 1.1, 30)
    reg_arr = [0]

    VAFs = []
    basic_tf_recreated = None
    if DEBUGGING:
        graph_data = None

    for reg in reg_arr:
        design_params = util.PlantDesignParams(
            num_order=num_order_discrete,
            denum_order=denum_order_discrete,
            better_cond_method=util.NORMALIZING,
            regularization=reg,
            sensor_data_column_names=None,
        )
        design_outcome = util.PlantDesignOutcome('test_plant')
        design_outcome.id_plant(sensor_data_test, design_params)

        if DEBUGGING:
            print(f'{design_outcome.train_perform.conditioning=}')

        temp_basic_tf_recreated = design_outcome.plant.delta_y_over_delta_u_c

        test_perf = util.PlantTestingPerformance()

        temp_graph_data: util.PlantGraphData = test_perf.compute_testing_performance(sensor_data_test, design_outcome.plant)
        
        VAFs.append(test_perf.VAF)

        if test_perf.VAF == max(VAFs):
            basic_tf_recreated = temp_basic_tf_recreated
            if DEBUGGING:
                graph_data = temp_graph_data

        # result_step_recreated: control.TimeResponseData = control.forced_response(input_to_torque_tf_recreated, T=t_step,U=prbs, X0=x0)

        # sensor_data_recreated = util.SensorData.from_timeseries(
        #     t = result_step_recreated.time, 
        #     r=result_step_recreated.inputs, 
        #     u=result_step_recreated.inputs, 
        #     y=result_step_recreated.outputs
        # )
        del design_outcome
        
    if DEBUGGING:
        if graph_data:
            graph_data._show_data(util.PlantGraphDataCommands(show=True), name="Test Graph")

        print(basic_tf.__repr__())
        if basic_tf_recreated:
            print(basic_tf_recreated.__repr__())
        print(f'{list(zip(reg_arr, VAFs))=}')

    assert(max(VAFs) >= 95 and _tf_equal(basic_tf, basic_tf_recreated))

test_correct_plant_deduced(_basic_tf())