import plant_util as util


str_test = "[-5.62482204e+00,2.36710609e+02,6.21409713e+04]|[1.00000000e+00,1.17973211e+02,5.05870774e+04,1.19886590e+04]"

tf_a = util.tf_decoder(str_test)

print(tf_a)

print(tf_a.poles())
print(tf_a.zeros())