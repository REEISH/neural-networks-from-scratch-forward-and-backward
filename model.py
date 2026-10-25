"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
        x_float = np.array(x, dtype = np.float64)
        grads = np.zeros_like(x_float)
        if x_float.size == 0:
            return grads
        it = np.nditer(x_float, flags=['multi_index'], op_flags = ['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_value = x_float[idx]
            x_float[idx] = old_value + eps
            f_pos = f(x_float)
            x_float[idx] = old_value - eps
            f_neg = f(x_float)
            grads[idx] = (f_pos - f_neg)/(2.0*eps)
            x_float[idx] = old_value
            it.iternext()
        return grads

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    diff = np.abs(analytic_grad - numeric_grad)
    denom = np.maximum(np.maximum(np.abs(analytic_grad),np.abs(numeric_grad)), tol)
    res = diff/denom
    return np.max(res)

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    # TODO: your approach here
    W, b = weight_init_fn(in_dim, out_dim)
    params = {'W':W, 'b':b}

    def forward(x):
      y = np.dot(x, params['W']) + params['b'] 
      cache = x
      return y, cache
    
    def backward(dout, cache):
      x = cache
      dx = np.dot(dout, params['W'].T)
      dW = np.dot(x.T, dout)
      db = np.sum(dout, axis = 0)
      grads = {'W':dW, 'b':db}
      return dx, grads

    return {'params':params, 'forward':forward, 'backward': backward}

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

