# -*- coding: utf-8 -*-
"""som1D.ipynb

Automatically generated from the notebook-style version.

# Self-Organizing Maps (1D chain)
"""

'''

SOM : Self-organizing Maps (1D chain)

Date: 4/25/2026
Author: Yoonsuck Choe

'''

"""# Utility Code: weight initialization and plotting"""

'''

Utility code : weight initialization, plotting, etc.

'''


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#---------------------------------------------------------------------
def som_w_init(n, r):
#---------------------------------------------------------------------
  '''
  W_init : init weight vectors for 1D chain
   arguments:
      n : SOM chain length
      r : randomness (0 <= r <= 1)
   returns: two vectors [W0, W1], each as shape (n,)
  '''

  a_idx = np.array(range(0, n))

  # Start with a diagonal chain in input space, then add noise.
  W0 = a_idx / n
  W1 = a_idx / n

  W0 = W0 * (1 - r) + (np.random.rand(n) * r)
  W1 = W1 * (1 - r) + (np.random.rand(n) * r)

  return [W0, W1]


#---------------------------------------------------------------------
def som_w_plot(W0, W1, ax=None):
#---------------------------------------------------------------------
  '''
  W_plot : plot SOM chain in input space
  arguments:
    W0 : w0 values
    W1 : w1 values
  '''

  if ax is None:
    ax = plt.gca()

  ax.plot(W0, W1, "-o", markersize=3)


#---------------------------------------------------------------------
def som_neigh(u, v, s):
#---------------------------------------------------------------------
  '''
  1D neighborhood function on chain index distance.

  arguments:
    u : BMU index (int)
    v : neuron index (int)
    s : std of gaussian neighbor (~ neighborhood radius)
  returns: neighborhood function value
  '''

  dist = abs(u - v)
  s_eff = max(float(s), 1e-8)
  g = 1.0/(s_eff * math.sqrt(2*math.pi)) * math.exp(-dist*dist/(2*s_eff*s_eff))

  return g


#---------------------------------------------------------------------
def som_bmu(W0, W1, x):
#---------------------------------------------------------------------
  '''
  som_bmu : find best matching unit index for input x.

  arguments:
    W0 : w_0 vector (for 1D chain)
    W1 : w_1 vector
    x  : 2D input vector
  returns:
    bmu : index of best matching unit
  '''

  diff0 = W0 - x[0]
  diff1 = W1 - x[1]
  dist = np.multiply(diff0, diff0) + np.multiply(diff1, diff1)

  bmu = int(dist.argmin())
  return bmu


#---------------------------------------------------------------------
def som_plot_inp(inp):
#---------------------------------------------------------------------
  '''
  som_plot_inp : plot input samples

  arguments:
    inp : input matrix (m x 2), m samples
  '''

  fig = plt.figure()
  ax = fig.add_subplot()
  ax.set_aspect("equal")
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  ax.invert_yaxis()
  ax.plot(inp[:, 0], inp[:, 1], ".")


#---------------------------------------------------------------------
def som_plot_model(k, x, bmu, W0, W1, neigh_vec, alpha, radius):
#---------------------------------------------------------------------
  '''
  som_plot_model: plots the chain and various information about the current model.

  argument:
    k         : current iteration #
    x         : input vector (1x2)
    bmu       : best matching unit index
    W0        : w_0 vector
    W1        : w_1 vector
    neigh_vec : n-sized neighborhood values for current BMU
    alpha     : current learning rate
    radius    : current neighborhood radius
  '''

  fig, axs = plt.subplots(1, 2, figsize=(12, 4))

  axs[0].plot(neigh_vec)
  axs[0].set_title("Neighborhood (1D chain)")
  axs[0].set_xlabel("Neuron index")
  axs[0].set_ylabel("g")

  som_w_plot(W0, W1, ax=axs[1])
  axs[1].plot(x[0], x[1], "r*", markersize=12)
  axs[1].plot(W0[bmu], W1[bmu], "mx", markersize=12, markeredgewidth=2)
  axs[1].set_aspect("equal")
  axs[1].set_xlim([0, 1])
  axs[1].set_ylim([0, 1])
  axs[1].invert_yaxis()
  axs[1].set_title("SOM Chain in Input Space")

  plt.tight_layout()
  print("[Iter "+str(k)+"]\t Input = "+str(x)+";\t alpha ="+str(alpha)+";\t radius ="+str(radius))


#---------------------------------------------------------------------
def som_animate(history, interval_ms=700):
#---------------------------------------------------------------------
  '''
  Animate saved 1D SOM snapshots after training.

  arguments:
    history     : list of snapshot dicts
    interval_ms : frame interval in milliseconds
  returns:
    animation object (or None if history is empty)
  '''

  if len(history) == 0:
    return None

  fig, axs = plt.subplots(1, 2, figsize=(12, 5))
  neigh_max = max(np.max(frame["neigh_vec"]) for frame in history)
  neigh_max = max(neigh_max, 1e-12)

  def _update(frame_idx):
    frame = history[frame_idx]

    axs[0].clear()
    axs[1].clear()

    axs[0].plot(frame["neigh_vec"])
    axs[0].set_ylim([0.0, neigh_max * 1.05])
    axs[0].set_title("Neighborhood @ iter " + str(frame["k"]))
    axs[0].set_xlabel("Neuron index")
    axs[0].set_ylabel("g")

    som_w_plot(frame["W0"], frame["W1"], ax=axs[1])
    axs[1].plot(frame["x"][0], frame["x"][1], "r*", markersize=12)
    axs[1].plot(frame["W0"][frame["bmu"]], frame["W1"][frame["bmu"]], "mx", markersize=12, markeredgewidth=2)
    axs[1].set_aspect("equal")
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])
    axs[1].invert_yaxis()
    axs[1].set_title(
      "Input = " + str(frame["x"]) +
      "\nalpha=" + str(frame["alpha"]) +
      ", radius=" + str(frame["radius"])
    )

    return axs

  ani = FuncAnimation(fig, _update, frames=len(history), interval=interval_ms, repeat=True)
  plt.tight_layout()
  plt.show()
  return ani


#---------------------------------------------------------------------
# 1. TEST weight init and plotting
#---------------------------------------------------------------------
[W0, W1] = som_w_init(10, 0.03)
plt.figure()
som_w_plot(W0, W1)

[W0, W1] = som_w_init(10, 1.0)
plt.figure()
som_w_plot(W0, W1)

#---------------------------------------------------------------------
# 2. TEST neighborhood function
#---------------------------------------------------------------------
x = np.linspace(-5, 5, 100)
y = np.zeros(x.size)
std = 2.0
for i in range(0, y.size):
  y[i] = som_neigh(0, int(x[i]), std)

plt.figure()
plt.plot(x, y)

# 3. TEST input plotting
inp = np.random.rand(200, 2)
som_plot_inp(inp)

"""# SOM: Main algorithm"""

'''
SOM Algorithm
'''


#---------------------------------------------------------------------
def som(n, inp, alpha, radius, alpha_reduce_rate, radius_reduce_rate, w_rand, num_loop):
#---------------------------------------------------------------------
  '''
  SOM with 1D chain.

  arguments:
    n : chain size
    inp : input data (m x 2), m 2D samples.
    alpha = 0.3      # initial learning rate
    radius = 4.0     # initial neighborhood radius
    alpha_reduce_rate = 1   # alpha reduction rate (>1)
    radius_reduce_rate = 2  # neighborhood radius reduction rate (>1)
    w_rand = 0.7     # weight setup randomness
    num_loop = 2000  # number of iterations to train
  '''

  num_inp = inp[:, 0].size

  # 1. init weights
  [W0, W1] = som_w_init(n, w_rand)

  # 2. loop
  history = []
  snapshot_interval = int(np.floor(num_loop/10))
  snapshot_interval = max(snapshot_interval, 1)

  for k in range(1, num_loop):

    # 2.1 randomly draw input from input set
    inp_idx = np.random.randint(num_inp)
    x = inp[inp_idx]

    # 2.2 find best matching unit
    bmu = som_bmu(W0, W1, x)

    # 2.3 update weights
    neigh_vec = np.zeros(n)  # for plotting

    for i in range(0, n):
      neigh_val = som_neigh(bmu, i, radius)
      neigh_vec[i] = neigh_val

      W0[i] = W0[i] + alpha * neigh_val * (x[0] - W0[i])
      W1[i] = W1[i] + alpha * neigh_val * (x[1] - W1[i])

    # 2.4. Update alpha and neighborhood radius
    alpha = alpha * (1 - 1/(num_loop/alpha_reduce_rate))
    radius = radius * (1 - 1/(num_loop/radius_reduce_rate))

    # 2.5 Save snapshot for animation playback after training
    if (np.mod(k, snapshot_interval) == 1) or (k == num_loop - 1):
      print("[Iter "+str(k)+"]\t Input = "+str(x)+";\t alpha ="+str(alpha)+";\t radius ="+str(radius))
      history.append(
        {
          "k": k,
          "x": np.array(x, copy=True),
          "bmu": int(bmu),
          "W0": np.array(W0, copy=True),
          "W1": np.array(W1, copy=True),
          "neigh_vec": np.array(neigh_vec, copy=True),
          "alpha": alpha,
          "radius": radius,
        }
      )

  print("Training complete. Showing animation...")
  return som_animate(history)


##################################
# TEST
##################################

# config
n = 200       # 1D chain length
num_inp = 1000
test_select = 1

# set up input
if (test_select == 1):
  # 1. random
  test_inp = np.random.rand(num_inp, 2)
elif (test_select == 2):
  # 2. gaussian
  test_inp = (np.random.normal(0, 5, [num_inp, 2]))
  test_inp = (test_inp-test_inp.min())/(test_inp.max()-test_inp.min())
elif (test_select == 3):
  # 3. hole in the middle
  test_inp = np.random.rand(num_inp, 2)
  test_inp = test_inp[np.multiply(test_inp[:,0]-0.5,test_inp[:,0]-0.5) + np.multiply(test_inp[:,1]-0.5,test_inp[:,1]-0.5) > 0.1,:]

# plot input
som_plot_inp(test_inp)

# Run SOM
'''
Example parameters:
alpha = 0.3      # initial learning rate
radius = 70.0    # initial neighborhood radius
alpha_reduce_rate = 1.2  # alpha reduction rate (>1)
radius_reduce_rate = 5   # neighborhood radius reduction rate (>1)
w_rand = 0.7     # weight setup randomness
num_loop = 100000  # number of iterations to train
'''
anim = som(n, test_inp, alpha=0.3, radius=70.0, alpha_reduce_rate=1.2, radius_reduce_rate=5, w_rand=0.7, num_loop=100000)

