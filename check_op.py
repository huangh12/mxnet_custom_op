from mxnet.test_utils import check_numeric_gradient
from focal_loss import *
import numpy as np

data = np.random.random((10, 1))
label = np.random.randint(-1, 2, size=(10,1))
print(data, label)

input_values = [data, label]
check_numeric_gradient(sym=mx.sym.Custom(op_type='focal_loss', use_ignore=True, normalize='valid'),
                       location=input_values,
                       grad_nodes=['custom0_data'])

# sym = mx.sym.smooth_l1(scalar=1.0)
# check_numeric_gradient(sym=sym,
#                        location=[data, label],
#                        grad_nodes=['softmaxoutput0_data', 'softmaxoutput0_label'])


# sym = mx.sym.SoftmaxOutput(normalization='valid', use_ignore=True, ignore_label=-1, )

# check_numeric_gradient(sym=sym,
#                        location=[data, label],
#                        grad_nodes=['softmaxoutput0_data', 'softmaxoutput0_label'])                       