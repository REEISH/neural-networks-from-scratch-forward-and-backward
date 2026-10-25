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

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    # TODO: your approach here
    forward = model['forward']
    backward = model['backward']
    logits, caches = forward(x)
    loss, d_logits = loss_fn(logits, y)
    dx, param_grads = backward(d_logits, caches)
    return float(loss), param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    # TODO: your approach here
    def _flatten(container, prefix = ""):
      if isinstance(container, dict):
        for key, val in container.items():
          yield from _flatten(val, f"{prefix}_{key}" if prefix else str(k))
      elif isinstance(container, list):
        for idx, val in enumerate(container):
          yield from _flatten(val, f"{prefix}_{idx}" if prefix else str(idx))
      elif isinstance(container, np.ndarray):
        yield prefix, container

    flat_params = dict(_flatten(params))

    state = {}

    if kind == 'momentum':
      state['v'] = {k:np.zeros_like(p) for k, p in flat_params.items()}
      state['beta'] = 0.9
    elif kind == 'adam':
      state['t'] = 0
      state['m'] = {k:np.zeros_like(p) for k, p in flat_params.items()}
      state['v'] = {k:np.zeros_like(p) for k, p in flat_params.items()}
      state['beta1'] = 0.9
      state['beta2'] = 0.999
      state['eps'] = 1e-8
    
    def step(grads):
      flat_grads = dict(_flatten(grads))

      if kind == 'sgd':
        for k, param in flat_params.items():
          if k in flat_grads and flat_grads[k] is not None:
            param -= lr*flat_grads[k]
      elif kind == 'momentum':
        beta = state['beta']
        for k, param in flat_params.items():
          if k in flat_grads and flat_grads[k] is not None:
            state['v'][k] = beta * state['v'][k] + flat_grads[k]
            param -= lr*state['v'][k]
      elif kind == 'adam':
        state['t'] += 1
        t = state['t']
        b1, b2, eps = state['beta1'], state['beta2'], state['eps']
        for k, param in flat_params.items():
          if k in flat_grads and flat_grads[k] is not None:
            g = flat_grads[k]
            state['m'][k] = b1 * state['m'][k] + (1.0 - b1) * g
            state['v'][k] = b2 * state['v'][k] + (1.0 - b2) * (g ** 2)
            m_hat = state['m'][k] / (1.0 - b1 ** t)
            v_hat = state['v'][k] / (1.0 - b2 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
      else:
            raise ValueError(f"Unknown optimizer variant: {kind}")


    return {'step':step}

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    # TODO: your approach here
    loss , param_grads = forward_backward(model, loss_fn, x_batch, y_batch)
    optimizer['step'](param_grads)
    return loss

# Step 11 - train
def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    # TODO: your approach here
    rng = np.random.default_rng(seed)
    num_samples = x.shape[0]
    history = []
    for epoch in range(epochs):
      permutation = rng.permutation(num_samples)
      x_shuffled = x[permutation]
      y_shuffled = y[permutation]
      epoch_losses = []

      for i in range(0, num_samples, batch_size):
        x_batch = x_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        batch_loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)
        epoch_losses.append(batch_loss)
      
      mean_epoch_loss = float(np.mean(epoch_losses))
      history.append(mean_epoch_loss)

    return history

# Step 12 - design_network
def design_network(input_dim, num_classes, seed=0):
    """Design and train a net that solves a nonlinear classification task.

    Inputs:
      input_dim: int, feature dimension
      num_classes: int, number of classes
      seed: int, RNG seed for reproducibility

    Returns:
      model: trained sequential model (forward/backward/params)
      metrics: dict with
        'accuracy': float >= 0.90 on an evaluation set,
        'x': np.ndarray (N, input_dim) eval features (N >= 50),
        'y': np.ndarray (N,) integer eval labels.
      The eval set (x, y) must not be linearly separable to high accuracy
      (< 0.82 for a linear classifier), and the model's true accuracy on
      it must match metrics['accuracy'] and be >= 0.90.
    """
    # TODO: your approach here
    rng = np.random.default_rng(seed)
    total_samples = 1000
    samples_per_class = total_samples // num_classes
    X_list = []
    y_list = []

    for c in range(num_classes):
      normal_vectors = rng.normal(0,1, size = (samples_per_class, input_dim))
      norms = np.linalg.norm( normal_vectors, axis = 1, keepdims = True)
      norms = np.where(norms == 0, 1e-5, norms)
      directions = normal_vectors / norms 
      radius = (c + 1) * 2.0 + rng.normal(0, 0.25, size=(samples_per_class, 1))
      X_list.append(directions * radius)
      y_list.append(np.full(samples_per_class, c))

    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)
    shuffled_indices = rng.permutation(len(y_all))
    X_all = X_all[shuffled_indices]
    y_all = y_all[shuffled_indices]
    X_train, X_eval = X_all[:800], X_all[800:]
    y_train, y_eval = y_all[:800], y_all[800:]

    hidden_dim = 128
    init_fn = lambda in_dim,out_dim: initialize_weights(in_dim, out_dim, scheme = "he")
    layer1 = make_dense(input_dim, hidden_dim, init_fn)
    act1   = make_activation('relu')
    layer2 = make_dense(hidden_dim, hidden_dim, init_fn)
    act2   = make_activation('relu')
    layer3 = make_dense(hidden_dim, num_classes, init_fn)
    model = make_sequential([layer1, act1, layer2, act2, layer3])

    loss_fn = make_loss("cross_entropy")
    optimizer = make_optimizer(model['params'], lr=0.01, kind='adam')

    _ = train(model, loss_fn, optimizer, X_train, y_train, epochs = 30, batch_size = 50, seed = seed)

    eval_logits, _ = model['forward'](X_eval)
    predictions = np.argmax(eval_logits, axis = 1)
    accuracy = float(np.mean(predictions == y_eval))

    metrics = {
        'accuracy': accuracy,
        'x': X_eval,
        'y': y_eval
    }

    return model, metrics

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

