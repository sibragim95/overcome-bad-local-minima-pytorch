# P_ReLU_DNN_all_params_Random_Normal_PyTorch.py

from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
import torch; import numpy as np
import matplotlib.pyplot as plt; import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

start_time = time.time()

train_steps = 50000; T = train_steps; batch_size = 1024
Dim1 = [1, 8, 8, 1]; Dim2 = [1, 15, 15, 1]; Dim3 = [1, 50, 50, 1]
Initializations = 300; M = Initializations; k = 2*(len(Dim1)-1)

# Realization function associated to ANNs
var_list, d, Rn = [], [], torch.randn
def Var(tensor, name='', trainable=True):
    return tensor.requires_grad_(trainable)

def realization(activation, nns):
    layers = []
    def layer(h0, h1, act):
        W = Var(Rn((M, h0, h1), device=device), 'weights')
        B = Var(Rn((M, 1, h1), device=device), 'biases')
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

# Activation; GD Optimizer; Target function
ReLU = torch.relu; Opt = torch.optim.SGD
def f(x):
    return torch.sin(3*x)

torch.manual_seed(5)

# Initialize networks
net1 = realization(ReLU, Dim1)
net2 = realization(ReLU, Dim2)
net3 = realization(ReLU, Dim3)

# PyTorch Optimizers attach directly to the variables
Opt1 = Opt(var_list[0:k], lr=0.0005)
Opt2 = Opt(var_list[k:2*k], lr=0.0005)
Opt3 = Opt(var_list[2*k:3*k], lr=0.0005)

# ANN parameter slices
par1 = var_list[0:k]
par2 = var_list[k:2*k]
par3 = var_list[2*k:3*k]

Grad1, p1, P1 = [], [], []
Grad2, p2, P2 = [], [], []
Grad3, p3, P3 = [], [], []

# Session Run equivalent
for i in range(T):
    # Monte Carlo Sampling & Risk dynamically calculated
    X = torch.rand((batch_size, 1), device=device)

    L1 = torch.mean((net1(X) - f(X.unsqueeze(0)))**2, 1)
    L2 = torch.mean((net2(X) - f(X.unsqueeze(0)))**2, 1)
    L3 = torch.mean((net3(X) - f(X.unsqueeze(0)))**2, 1)

    l_rate = 0.0005*(i+1)**(-1/10)
    for param_group in Opt1.param_groups: param_group['lr'] = l_rate
    for param_group in Opt2.param_groups: param_group['lr'] = l_rate
    for param_group in Opt3.param_groups: param_group['lr'] = l_rate

    Opt1.zero_grad(); L1.sum().backward(); Opt1.step()
    Opt2.zero_grad(); L2.sum().backward(); Opt2.step()
    Opt3.zero_grad(); L3.sum().backward(); Opt3.step()

    if (i%20==0):
        par1_v = [p.reshape(M, d_i).detach().cpu().numpy() for p, d_i in zip(par1, d[0:k])]
        par2_v = [p.reshape(M, d_i).detach().cpu().numpy() for p, d_i in zip(par2, d[k:2*k])]
        par3_v = [p.reshape(M, d_i).detach().cpu().numpy() for p, d_i in zip(par3, d[2*k:3*k])]

        g1_v = [p.grad.reshape(M, d_i).detach().cpu().numpy() for p, d_i in zip(par1, d[0:k])]
        g2_v = [p.grad.reshape(M, d_i).detach().cpu().numpy() for p, d_i in zip(par2, d[k:2*k])]
        g3_v = [p.grad.reshape(M, d_i).detach().cpu().numpy() for p, d_i in zip(par3, d[2*k:3*k])]

        parr1_v, gg1_v = np.concatenate(par1_v, 1), np.concatenate(g1_v, 1)
        parr2_v, gg2_v = np.concatenate(par2_v, 1), np.concatenate(g2_v, 1)
        parr3_v, gg3_v = np.concatenate(par3_v, 1), np.concatenate(g3_v, 1)

        p1.append(parr1_v); Grad1.append(np.linalg.norm(gg1_v)/(M**(1/2)))
        p2.append(parr2_v); Grad2.append(np.linalg.norm(gg2_v)/(M**(1/2)))
        p3.append(parr3_v); Grad3.append(np.linalg.norm(gg3_v)/(M**(1/2)))

for i in range(len(p1)-1):
    P1.append(np.linalg.norm(p1[i]-p1[-1])/(M**(1/2)))
    P2.append(np.linalg.norm(p2[i]-p2[-1])/(M**(1/2)))
    P3.append(np.linalg.norm(p3[i]-p3[-1])/(M**(1/2)))

Param1, Param2, Param3 = P1, P2, P3

# Calculate final risk levels
X_final = torch.rand((batch_size, 1), device=device)
graph1 = torch.mean((net1(X_final) - f(X_final.unsqueeze(0)))**2, 1).detach().cpu().numpy().flatten()
graph2 = torch.mean((net2(X_final) - f(X_final.unsqueeze(0)))**2, 1).detach().cpu().numpy().flatten()
graph3 = torch.mean((net3(X_final) - f(X_final.unsqueeze(0)))**2, 1).detach().cpu().numpy().flatten()

# ----------------------- Numerical Simulation -----------------------
font1 = {'fontname':'Perpetua', 'size':16}
font2 = {'fontname':'Perpetua', 'size':11}; splt = plt.subplot
gs = GridSpec(2, 2, width_ratios=[2, 0.75], height_ratios=[1, 1])
gs.update(top=1, right=1.8, wspace=0.17, hspace=0.35)
fig1, fig2, fig3 = splt(gs[:, 0]), splt(gs[0, 1]), splt(gs[1, 1])
common_bins=100
C1, B1 = np.histogram(graph1, bins=common_bins)
fig1.hist(B1[:-1], B1, weights=C1/M, color="gray", edgecolor="darkgray",
          label=r'$(\ell_0, \ell_1, \ell_2, \ell_3) =$'+format(tuple(Dim1)))
C2, B2 = np.histogram(graph2, bins=common_bins)
fig1.hist(B2[:-1], B2, weights=C2/M, color="orange", edgecolor="darkorange",
          label=r'$(\ell_0, \ell_1, \ell_2, \ell_3) =$'+format(tuple(Dim2)))
C3, B3 = np.histogram(graph3, bins=common_bins)
fig1.hist(B3[:-1], B3, weights=C3/M, color="teal", edgecolor="darkgreen",
          label=r'$(\ell_0, \ell_1, \ell_2, \ell_3) =$'+format(tuple(Dim3)))
fig1.yaxis.set_major_formatter(PercentFormatter(1))
fig1.set_xscale('log', base=10)
fig1.legend()
fig1.set_xlabel('Risk levels', fontdict=font1)
fig1.set_ylabel('Relative frequency of risk levels', fontdict=font1)
xx=np.linspace(1, T, len(Param1), dtype=int)
fig2.loglog(xx, Grad1[0:len(Grad1)-1], color='gray')
fig2.loglog(xx, Grad2[0:len(Grad2)-1], color='darkorange')
fig2.loglog(xx, Grad3[0:len(Grad3)-1], color='teal')
fig2.set_xlabel('Number of SGD training steps', fontdict=font2)
fig2.set_ylabel('Norm of the gradient', fontdict=font2)
fig3.semilogx(xx, Param1, color = 'gray')
fig3.semilogx(xx, Param2, color = 'darkorange')
fig3.semilogx(xx, Param3, color = 'teal')
fig3.set_xlabel('Number of SGD training steps', fontdict=font2)
fig3.set_ylabel(r'$\mathcal{L}^{2}\!$'+'-distance to limit point', fontdict=font2)
plt.savefig("Opt_DNN_SGD_8_15_50_ReLU_all_params_train.pdf", bbox_inches="tight")
plt.show()

end_time = time.time()
elapsed = end_time - start_time
print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
