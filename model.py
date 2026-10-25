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

# Step 2 - gradient_check (not yet solved)
# TODO: implement

# Step 3 - make_dense (not yet solved)
# TODO: implement

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

