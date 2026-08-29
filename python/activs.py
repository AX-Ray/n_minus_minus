import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

def sigmoid_f(x): 
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_b(x): 
    return x * (1.0 - x)

def relu_f(x): 
    return np.maximum(0.0, x)

def relu_b(x): 
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu_f(x, alpha=0.01): 
    return np.where(x > 0, x, alpha * x)

def leaky_relu_b(x, alpha=0.01): 
    return np.where(x > 0, 1.0, alpha)

def linear_f(x): 
    return x

def linear_b(x): 
    return np.ones_like(x)

def tanh_f(x): 
    return np.tanh(x)

def tanh_b(x): 
    return 1.0 - x * x

def selu_f(x, l=1.05070098735548, a=1.67326324235437): 
    return np.where(x >= 0, l * x, l * a * (np.exp(x) - 1.0))

def selu_b(x, l=1.05070098735548, a=1.67326324235437): 
    return np.where(x >= 0, l, x + l * a)

def swish_f(x): 
    return x / (1.0 + np.exp(-x))

def swish_b(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig + sig * (1.0 - x * sig)

def mish_f(x):
    sp = np.where(x > 20.0, x, np.log1p(np.exp(x)))
    return x * np.tanh(sp)

def mish_b(x):
    ex = np.exp(x)
    ex2 = ex * ex
    ex3 = ex2 * ex
    omega = 4.0 * (x + 1.0) + 4.0 * ex2 + ex3 + ex * (4.0 * x + 6.0)
    delta = 2.0 * ex + ex2 + 2.0
    return (ex * omega) / (delta * delta)

def lslu_f(x):
    swish = x / (1.0 + np.exp(-x))
    softplus = np.where(x > 20.0, x, np.log1p(np.exp(x)))
    return swish + softplus - np.log(2.0)

def lslu_b(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    d_swish = x * sig + sig * (1.0 - x * sig)
    d_softplus = sig
    return d_swish + d_softplus

def gelu_f(x):
    kSqrt2OverPi = 0.7978845608028654
    inner = kSqrt2OverPi * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))

def gelu_b(x):
    kSqrt2OverPi = 0.7978845608028654
    inner = kSqrt2OverPi * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    sech_sq = 1.0 - tanh_inner * tanh_inner
    d_inner = kSqrt2OverPi * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_sq * d_inner

def elu_f(x, alpha=1.0): 
    return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1.0))

def elu_b(x, alpha=1.0): 
    return np.where(x >= 0.0, 1.0, alpha * np.exp(x))

def softplus_f(x): 
    return np.where(x > 20.0, x, np.log1p(np.exp(x)))

def softplus_b(x): 
    return 1.0 / (1.0 + np.exp(-x))

def aplu_f(x, a=0.05, l=1.0, b=0.2, g=1.0):
    base = np.where(x >= 0.0, l * x, a * x)
    return base + b * np.tanh(g * x)

def aplu_b(x, a=0.05, l=1.0, b=0.2, g=1.0):
    d_base = np.where(x >= 0.0, l, a)
    t = np.tanh(g * x)
    d_tanh = b * g * (1.0 - t * t)
    return d_base + d_tanh

def gba_f(x, eps=0.05, beta=1.5, amp=0.25, center=2.0):
    beta_x = beta * x
    sp_centered = (np.log(1.0 + np.exp(beta_x)) - np.log(2.0)) / beta
    tanh_center = np.tanh(center)
    bridge = np.tanh(x + center) - tanh_center
    return eps * x + (1.0 - eps) * sp_centered + amp * bridge

def gba_b(x, eps=0.05, beta=1.5, amp=0.25, center=2.0):
    sig = 1.0 / (1.0 + np.exp(-beta * x))
    tanh_val = np.tanh(x + center)
    sech_sq = 1.0 - tanh_val * tanh_val
    return eps + (1.0 - eps) * sig + amp * sech_sq

activations = [
    ("Sigmoid", sigmoid_f, sigmoid_b, True),
    ("ReLU", relu_f, relu_b, False),
    ("LeakyReLU (alpha=0.1)", lambda x: leaky_relu_f(x, 0.1), lambda x: leaky_relu_b(x, 0.1), False),
    ("Linear", linear_f, linear_b, False),
    ("TanH", tanh_f, tanh_b, True),
    ("SELU", selu_f, selu_b, False),
    ("Swish", swish_f, swish_b, False),
    ("Mish", mish_f, mish_b, False),
    ("LSLU", lslu_f, lslu_b, False),
    ("GELU", gelu_f, gelu_b, False),
    ("ELU (alpha=1.0)", elu_f, elu_b, False),
    ("Softplus", softplus_f, softplus_b, False),
    ("APLU", aplu_f, aplu_b, False),
    ("GBA", lambda x: gba_f(x), lambda x: gba_b(x), False)
]

plt.style.use('seaborn-v0_8-grid' if 'seaborn-v0_8-grid' in plt.style.available else 'default')

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

activation_chunks = list(chunk_list(activations, 3))
num_plots = len(activation_chunks)

fig1, axes1 = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
fig1.suptitle("Activation Functions: Forward Pass", fontsize=16, fontweight='bold', y=0.99)

for idx, chunk in enumerate(activation_chunks):
    ax = axes1[idx]
    for name, f_func, _, _ in chunk:
        ax.plot(x, f_func(x), label=name, linewidth=3.5) 
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(loc="upper left")
    ax.set_ylabel("Output Y")

axes1[-1].set_xlabel("Input X")
fig1.tight_layout()

fig2, axes2 = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
fig2.suptitle("Activation Function Derivatives: Backward Pass", fontsize=16, fontweight='bold', y=0.99)

for idx, chunk in enumerate(activation_chunks):
    ax = axes2[idx]
    for name, f_func, b_func, use_activated_x in chunk:
        if use_activated_x:
            y_grad = b_func(f_func(x))
        else:
            y_grad = b_func(x)
        ax.plot(x, y_grad, label=name, linewidth=3.5) 
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(loc="upper left")
    ax.set_ylabel("Gradient dY/dX")

axes2[-1].set_xlabel("Input X")
fig2.tight_layout()

plt.show()