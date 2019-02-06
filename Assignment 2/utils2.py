import numpy as np

def square_hinge_loss(targets, outputs):
  targets = 2 * targets - 1
  x = 1 - targets * outputs
  x[x < 0] = 0
  x = np.sum(x * x)
  return x

def logistic_loss(targets, outputs):
  targets = 2 * targets - 1
  x = np.sum(np.log(1 + np.exp(-targets * outputs)))
  return x

def perceptron_loss(targets, outputs):
  targets = 2 * targets - 1
  x = - targets * outputs
  x[x < 0] = 0
  x = np.sum(x)
  return x

def L2_regulariser(weights):
    return np.sum(weights * weights)

def L4_regulariser(weights):
    return np.sum(weights * weights * weights * weights)

def square_hinge_grad(weights, inputs, targets, outputs):
  targets = 2 * targets - 1
  #print(np.shape(np.matmul(inputs,weights)))
  #print()
  x = 1 - np.matmul(inputs, weights) * targets
  #print(np.matmul(inputs, weights))
  x[x < 0] = 0
  #print(x)
  z = -2 * x[:, None] * inputs * targets[:, None]
  return np.sum(z, axis = 0)

def logistic_grad(weights,inputs, targets, outputs):
  targets = 2 * targets - 1
  x = -np.exp(-targets * outputs)/(1 + np.exp(-targets * outputs))
  # Write thee logistic loss loss gradient here
  #print(x)
  z = x[:, None] * inputs * targets[:, None]
  #print( np.sum(z, axis = 0))
  return np.sum(z, axis = 0)

def perceptron_grad(weights,inputs, targets, outputs):
  targets = 2 * targets - 1
  y = - targets * outputs
  y[y < 0] = 0
  y[y > 0] = 1
  x = -inputs * np.multiply(targets,y)[:, None]
  
  return np.sum(x, axis = 0)

def L2_grad(weights):
    # Write the L2 loss gradient here
    return 2 * (weights)

def L4_grad(weights):
    # Write the L4 loss gradient here
    return 4 * weights * weights * weights

loss_functions = {"square_hinge_loss" : square_hinge_loss, 
                  "logistic_loss" : logistic_loss,
                  "perceptron_loss" : perceptron_loss}

loss_grad_functions = {"square_hinge_loss" : square_hinge_grad, 
                       "logistic_loss" : logistic_grad,
                       "perceptron_loss" : perceptron_grad}

regularizer_functions = {"L2": L2_regulariser,
                         "L4": L4_regulariser}

regularizer_grad_functions = {"L2" : L2_grad,
                              "L4" : L4_grad}
