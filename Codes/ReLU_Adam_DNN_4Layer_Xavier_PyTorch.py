from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
import torch; import numpy as np
import matplotlib.pyplot as plt; import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

start_time = time.time()

train_steps = 10000; T = train_steps; batch_size = 1024
Initializations = 300; M = Initializations; Dim = [1, 50, 50, 1]

# Realization function associated to deep ANNs
var_list, d, Rn = [], [], torch.randn
def Var(tensor, name='', trainable=True):
    return tensor.requires_grad_(trainable)

def realization(activation, nns):
    layers = []
    def layer(h0, h1, act):
# Initialize ANN parameters: Xavier normal
        W = Var((math.sqrt(2/(h0+h1))*Rn((M, h0, h1))).to(device), 'weights')
        B = Var(torch.zeros((M, 1, h1), device=device), 'biases')
        var_list.append(W); d.append(h0*h1)
        var_list.append(B); d.append(h1)
        layers.append((W, B, act))

    for i in range(len(nns)-2):
        layer(nns[i], nns[i+1], activation)
    layer(nns[len(nns)-2], nns[len(nns)-1], lambda x: x)

    def forward(x):
        y = x.unsqueeze(0)
        for W, B, act in layers:
            y = act(torch.matmul(y, W)+B)
        return y

    return forward

# Activation; Adam Optimizer; Target function
ReLU = torch.relu; Opt = torch.optim.Adam
def f(x):
    return x**2 + 2*x

torch.manual_seed(5)

# Initialize network
net = realization(ReLU, Dim)

# PyTorch Optimizer attaches directly to the variables
Opt1 = Opt(var_list, lr=0.01, betas=(0.9, 0.999), eps=1e-08)

# ANN parameter slices
par = var_list

Grad, p, P = [], [], []

# Session Run equivalent
for i in range(T):
    # Monte Carlo Sampling & Risk dynamically calculated
    X = torch.rand((batch_size, Dim[0]), device=device)

    L = torch.mean((net(X) - f(X.unsqueeze(0)))**2, 1)

    l_rate = 0.01*0.5**(int((i+1)/500))
    for param_group in Opt1.param_groups: param_group['lr'] = l_rate

    Opt1.zero_grad(); L.sum().backward(); Opt1.step()

    if (i%10==0):
        par_v = [p0.reshape(M, d_i).detach().cpu().numpy() for p0, d_i in zip(par, d)]
        g_v = [p0.grad.reshape(M, d_i).detach().cpu().numpy() for p0, d_i in zip(par, d)]

        parr_v, gg_v = np.concatenate(par_v, 1), np.concatenate(g_v, 1)

        p.append(parr_v); Grad.append(np.linalg.norm(gg_v)/(M**(1/2)))

for i in range(len(p)-1):
    P.append(np.linalg.norm(p[i]-p[-1])/(M**(1/2)))

Param = P

# Calculate final risk levels
X_final = torch.rand((batch_size, Dim[0]), device=device)
graph = torch.mean((net(X_final) - f(X_final.unsqueeze(0)))**2, 1).detach().cpu().numpy().flatten()

#----------------------- Numerical Simulation -----------------------
font1 = {'fontname':'Perpetua', 'size':16}
font2 = {'fontname':'Perpetua', 'size':11}; splt = plt.subplot
gs = GridSpec(2, 2, width_ratios=[2, 0.75], height_ratios=[1, 1])
gs.update(top=1, right=1.8, wspace=0.17, hspace=0.35)
fig1, fig2, fig3 = splt(gs[:, 0]), splt(gs[0, 1]), splt(gs[1, 1])
C, B = np.histogram(graph, bins=80)
fig1.hist(B[:-1], B, weights=C/M, color="teal", edgecolor="darkgreen")
fig1.yaxis.set_major_formatter(PercentFormatter(1))
fig1.set_xlabel('Risk levels', fontdict=font1)
fig1.set_ylabel('Relative frequency of risk levels', fontdict=font1)
xx=np.linspace(1, T, len(Param), dtype=int)
fig2.loglog(xx, Grad[0:len(Grad)-1], color='teal')
gm=int(np.log10(np.min(Grad))); gM=int(np.log10(np.max(Grad)))
fig2.set_yticks(np.logspace(gm, gM, num=gM-gm+1))
fig2.set_xlabel('Number of SGD training steps', fontdict=font2)
fig2.set_ylabel('Norm of the gradient', fontdict=font2)
fig3.semilogx(xx, Param, color = 'teal')
fig3.set_xlabel('Number of SGD training steps', fontdict=font2)
fig3.set_ylabel(r'$\mathcal{L}^{2}\!$'+'-distance to limit point', fontdict=font2)
plt.savefig("ReLU_DNN_Adam_1_50_50_1_Xavier.pdf", bbox_inches="tight")
plt.show()

end_time = time.time()
elapsed = end_time - start_time
print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")