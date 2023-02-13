import pytest
import control
from sysid import plant_util as util

def test_tf_encoder_decoder():

    s = control.tf('s')

    tf_test = (-5.62482204e+00*s**2 + 2.36710609e+02*s + 6.21409713e+04) / (1.00000000e+00*s**3 + 1.17973211e+02*s**2 + 5.05870774e+04*s + 1.19886590e+04)
    
    str_test = util.tf_encoder(tf_test)

    tf_post = util.tf_decoder(str_test)

    assert(tf_test.__repr__() == tf_post.__repr__())


test_tf_encoder_decoder()

