from PCGP.deep_tensor import PCGP_Builder_pytorch
import sympy as sp


def B(D, x):
    a = sp.Symbol("a")
    (d_dx,) = D
    #return sp.matrices.Matrix([[d_dx], [a]])
    return sp.matrices.Matrix([[1], [d_dx + a]])
    #return sp.matrices.Matrix([[1]])#add your B-matrix here


builder1 = PCGP_Builder_pytorch()
builder1.add_kernel(B,  number_of_input_dimensions=1, shared_base_kernel = True)
builder1.write("kernel_f1")
#f1 = 3*e^-(ax)
