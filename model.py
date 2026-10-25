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

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    # TODO: your approach here
    def forward(x):
      cache = x
      if kind == 'relu':
        y = np.maximum(0,x)
      elif kind == 'sigmoid':
        y = 1.0/(1.0 + np.exp(-np.clip(x, -500, 500)))
      elif kind == 'tanh':
        y = np.tanh(x)
      else:
        raise ValueError(f"Unknown activation type: {kind}") 
      return y, cache
  
    def backward(dout, cache):
      x = cache
      if kind == 'relu':
        dx = dout*(x > 0)
      elif kind == 'sigmoid':
        s = 1.0/(1.0 + np.exp(-np.clip(x, -500, 500)))
        dx = dout*(s)*(1 - s)
      elif kind == 'tanh':
        t = np.tanh(x)
        dx = dout*(1.0 - t**2)
      else:
        raise ValueError(f"Unknown activation type: {kind}")
      grads = {}
      return dx, grads
    return {
      'params':{},
      'forward':forward,
      'backward':backward
          }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    # TODO: your approach here
    W = np.random.randn(in_dim, out_dim )*np.sqrt((2.0 if scheme == 'he' else 1.0)/in_dim)
    return W, np.zeros(out_dim)

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    # TODO: your approach here
    def loss_fn(logits, labels):
      batch_dim = logits.shape[0]
      logits_shifted = logits - np.max(logits, axis = 1, keepdims = True)
      exp_logits = np.exp(logits_shifted)
      probs = exp_logits/np.sum(exp_logits, axis = 1, keepdims = True)
      correct_class_probs = probs[np.arange(batch_dim), labels]
      loss = -np.mean(np.log(np.clip(correct_class_probs, 1e-15, 1.0)))
      d_logits = probs.copy()
      d_logits[np.arange(batch_dim), labels] -= 1.0
      d_logits /= batch_dim
      return float(loss), d_logits
    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    # TODO: your approach here
    params = [layer['params'] for layer in layers]     

    def forward(x):
      caches = []
      curr = x
      for layer in layers:
        curr, layer_cache = layer['forward'](curr)
        caches.append(layer_cache)
      return curr, caches

    def backward(dout, caches):
      grad_list = [None]*len(layers)
      current_dout = dout
      for i in reversed(range(len(layers))):
        layer = layers[i]
        layer_cache = caches[i]
        current_dout, layer_grads = layer['backward'](current_dout, layer_cache)
        grad_list[i] = layer_grads
      dx = current_dout
      return dx, grad_list

    return {
      'params':params,
      'forward':forward,
      'backward':backward
    }

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

