
import torch



def kernel_0(x1, x2, parameters, structure):
    xx = x1[:, :-1]
    yy = x2[:, :-1]

    K = torch.zeros((x1.shape[0], x2.shape[0]), dtype=x1.dtype, device=x1.device)
    lengthscale = parameters['lengthscale']
    amplitude = parameters['amplitude']
    a = parameters['a']
    def k00(x, y):
        return amplitude*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)
    def k01(x, y):
        #print(torch.Tensor.size(a))
        #print(torch.Tensor.size(lengthscale))
        #print(torch.Tensor.size(y[...,0]))
        #print(torch.Tensor.size(x[...,0]))
        return amplitude*(a*lengthscale + x[...,0] - y[...,0])*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale
    def k10(x, y):
        return amplitude*(a*lengthscale - x[...,0] + y[...,0])*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale
    def k11(x, y):
        return amplitude*(a**2*lengthscale**2 + lengthscale - (x[...,0] - y[...,0])**2)*torch.exp(-1/2*(x[...,0] - y[...,0])**2/lengthscale)/lengthscale**2
    function_grid = [
        [k00, k01],
        [k10, k11],
    ]
    for i, idxs1 in enumerate(structure.x1):
        for j, idxs2 in enumerate(structure.x2):
            x_block = xx[idxs1]
            y_block = yy[idxs2]
            block = function_grid[i][j](x_block[:, None, :], y_block[None, :, :])
            K[idxs1.long()[:, None], idxs2.long()[None, :]] = block
    print(K.shape)
    return K


def kernel(x1, x2, parameters, structure):
    k = (
        kernel_0(x1, x2, parameters, structure)
        )
    return k

kernel = torch.compile(kernel, backend="eager")