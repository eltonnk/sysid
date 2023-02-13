import pytest
import control
import numpy as np
from sysid import plant_util as util
from sysid import d2c

def test_tf_encoder_decoder():
    s: control.TransferFunction = control.tf('s')

    tf_test: control.TransferFunction = (-5.62482204e+00*s**2 + 2.36710609e+02*s + 6.21409713e+04) / (1.00000000e+00*s**3 + 1.17973211e+02*s**2 + 5.05870774e+04*s + 1.19886590e+04)
    
    str_test = util.tf_encoder(tf_test)

    tf_post = util.tf_decoder(str_test)

    assert(tf_test.__repr__() == tf_post.__repr__()) # best we can do since __eq__() is not implented in control




def test_correct_plant_deduced():
    s: control.TransferFunction = control.tf('s')

    # System used as an example is a brushed DC motor

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

    _actuatorMaxTorque = 2

    # Generate input signal (PRBS)
    id_seq_lenght=20.0
    one_bit_period=0.5
    _CTRL_PERIOD = 200
    _samplingPeriod = _CTRL_PERIOD * 1e-6

    N = int((1/_samplingPeriod)*id_seq_lenght)
    oneBitN = int((1/_samplingPeriod)*one_bit_period)

    n = 0
    outputMem = 0.0

    start = 2
    a = start
    newbit = 0

    prbs = np.zeros((N,))
    t_step = np.zeros((N,))
    t_now = 0
    for n in range(N):
        if n % oneBitN == 0 :
            newbit = (((a >> 6) ^ (a >> 5)) & 1)
            a = ((a << 1) | newbit) & 0x7f
            outputMem = ((2.0 * newbit) - 1.0) * _actuatorMaxTorque
        prbs[n] = outputMem
        t_step[n] = t_now
        t_now += _samplingPeriod


    dt = _samplingPeriod
    x0 = np.array([0, 0]) # 2 states since 2nd order denominator for input_to_torque_tf
 
    result_step: control.TimeResponseData = control.forced_response(input_to_torque_tf, T=t_step,U=prbs, X0=x0)

    sensor_data = util.SensorData.from_timeseries(
        t = result_step.time, 
        r=result_step.inputs, 
        u=result_step.inputs, 
        y=result_step.outputs
    )

    # Uncomment code below to visualize step response result
    sensor_data.plot()
    import matplotlib.pyplot as plt
    
    design_outcome = util.PlantDesignOutcome('test_plant')

    input_to_torque_tf_discrete: control.TransferFunction = control.c2d(input_to_torque_tf, Ts=dt)

    design_params = util.PlantDesignParams(
        num_order=len(input_to_torque_tf_discrete.num[0][0])-1,
        denum_order=len(input_to_torque_tf_discrete.den[0][0])-1,
        better_cond_method=util.NORMALIZING,
        regularization=0.0,
        sensor_data_column_names=None,
    )
    design_outcome.id_plant(sensor_data, design_params)

    input_to_torque_tf_recreated = design_outcome.plant.delta_y_over_delta_u_c

    result_step: control.TimeResponseData = control.forced_response(input_to_torque_tf_recreated, T=t_step,U=prbs, X0=x0)

    sensor_data = util.SensorData.from_timeseries(
        t = result_step.time, 
        r=result_step.inputs, 
        u=result_step.inputs, 
        y=result_step.outputs
    )
    
    sensor_data.plot()
    plt.show()

    print(input_to_torque_tf.__repr__())
    print(input_to_torque_tf_recreated.__repr__())

test_correct_plant_deduced()