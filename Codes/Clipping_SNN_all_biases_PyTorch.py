from matplotlib.ticker import PercentFormatter
from matplotlib.gridspec import GridSpec
import torch; import numpy as np
import matplotlib.pyplot as plt

train_steps = 10000; T = train_steps; batch_size = 1024
Initializations = 300; M = Initializations; h1 = 10; h2 = 100; h3 = 1000

# Realization function associated to shallow ANNs
var_list, d, Rn = [], [], torch.randn
def Var(tensor, name='', trainable=True):
    return tensor.requires_grad_(trainable)

# We drop 'x' from the arguments so it builds the weights once,
# and returns a lambda function that acts exactly like your TF graph!
def realization(activation, h):
# Initialize ANN parameters: standard normal like
    w = Var((h**(3))*torch.abs(Rn((M, 1, h))), name='inner_weights', trainable=False)
    b = Var((h**(3))*Rn((M, 1, h)), name='inner_biases', trainable=True)
    v = Var((h**(-7/8))*torch.abs(Rn((M, h, 1))), name='outer_weights', trainable=False)
    c = Var(Rn((M, 1, 1)), name='outer_bias', trainable=True)
    var_list.append(b); d.append(h)
    var_list.append(c); d.append(1); mult = torch.matmul
    
    # Return the forward pass graph
    return lambda x: c + mult(activation(b + mult(x.unsqueeze(0), w)), v)

# Activation; GD Optimizer; Target function
def clipping(x):
    return torch.clamp(x, 0, 1)
Opt = torch.optim.SGD
def f(x):
    return x**(1/4)

torch.manual_seed(5)

# Initialize networks (weights are created and stored here once)
net1 = realization(clipping, h1)
net2 = realization(clipping, h2)
net3 = realization(clipping, h3)

# PyTorch Optimizers attach directly to the variables
Opt1 = Opt(var_list[0:2], lr=0.01)
Opt2 = Opt(var_list[2:4], lr=0.01)
Opt3 = Opt(var_list[4:6], lr=0.01)

# ANN parameter slices
par1 = var_list[0:2]
par2 = var_list[2:4]
par3 = var_list[4:6]

Grad1, p1, P1 = [], [], []
Grad2, p2, P2 = [], [], []
Grad3, p3, P3 = [], [], []

# Session Run (Training Loop)
for i in range(T):
    # --- 1-to-1 Math: Monte Carlo Sampling & Risk dynamically calculated ---
    X = torch.rand((batch_size, 1))
    
    L1 = torch.mean((net1(X) - f(X.unsqueeze(0)))**2, 1)
    L2 = torch.mean((net2(X) - f(X.unsqueeze(0)))**2, 1)
    L3 = torch.mean((net3(X) - f(X.unsqueeze(0)))**2, 1)
    # -----------------------------------------------------------------------

    l_rate = 0.01*0.5**(int((i+1)/150))
    for param_group in Opt1.param_groups: param_group['lr'] = l_rate
    for param_group in Opt2.param_groups: param_group['lr'] = l_rate
    for param_group in Opt3.param_groups: param_group['lr'] = l_rate

    # Calculate gradients and step the optimizer
    Opt1.zero_grad(); L1.sum().backward(); Opt1.step()
    Opt2.zero_grad(); L2.sum().backward(); Opt2.step()
    Opt3.zero_grad(); L3.sum().backward(); Opt3.step()

    if (i%20==0):
        # Fetch gradients directly from .grad and format for storage
        par1_v = [p.reshape(M, d_i).detach().numpy() for p, d_i in zip(par1, d[0:2])]
        par2_v = [p.reshape(M, d_i).detach().numpy() for p, d_i in zip(par2, d[2:4])]
        par3_v = [p.reshape(M, d_i).detach().numpy() for p, d_i in zip(par3, d[4:6])]
        
        g1_v = [p.grad.reshape(M, d_i).detach().numpy() for p, d_i in zip(par1, d[0:2])]
        g2_v = [p.grad.reshape(M, d_i).detach().numpy() for p, d_i in zip(par2, d[2:4])]
        g3_v = [p.grad.reshape(M, d_i).detach().numpy() for p, d_i in zip(par3, d[4:6])]
        
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

# Calculate final risk levels with a fresh batch to match TF session close
X_final = torch.rand((batch_size, 1))
graph1 = torch.mean((net1(X_final) - f(X_final.unsqueeze(0)))**2, 1).detach().numpy().flatten()
graph2 = torch.mean((net2(X_final) - f(X_final.unsqueeze(0)))**2, 1).detach().numpy().flatten()
graph3 = torch.mean((net3(X_final) - f(X_final.unsqueeze(0)))**2, 1).detach().numpy().flatten()

#----------------------- Numerical Simulation -----------------------
font1 = {'fontname':'Perpetua', 'size':16}
font2 = {'fontname':'Perpetua', 'size':11}; splt = plt.subplot
gs = GridSpec(2, 2, width_ratios=[2, 0.75], height_ratios=[1, 1])
gs.update(top=1, right=1.8, wspace=0.17, hspace=0.35)
fig1, fig2, fig3 = splt(gs[:, 0]), splt(gs[0, 1]), splt(gs[1, 1])
C1, B1 = np.histogram(graph1, bins=40)
fig1.hist(B1[:-1], B1, weights=C1/M, color="gray", edgecolor="darkgray",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple([1,h1,1])))
C2, B2 = np.histogram(graph2, bins=40)
fig1.hist(B2[:-1], B2, weights=C2/M, color="orange", edgecolor="darkorange",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple([1,h2,1])))
C3, B3 = np.histogram(graph3, bins=40)
fig1.hist(B3[:-1], B3, weights=C3/M, color="teal", edgecolor="darkgreen",
          label=r'$(\ell_0, \ell_1, \ell_2) =$'+format(tuple([1,h3,1])))
fig1.yaxis.set_major_formatter(PercentFormatter(1))
fig1.set_xscale('log', base=10)
fig1.legend()
fig1.set_xlabel('Risk levels', fontdict=font1)
fig1.set_ylabel('Relative frequency of risk levels', fontdict=font1)
xx=np.linspace(1, T, len(Param1), dtype=int)
fig2.loglog(xx, Grad1[0:len(Grad1)-1], color='gray')
fig2.loglog(xx, Grad2[0:len(Grad2)-1], color='darkorange')
fig2.loglog(xx, Grad3[0:len(Grad3)-1], color='teal')
gm=int(np.log10(np.min([Grad1, Grad2, Grad3])))
gM=int(np.log10(np.max([Grad1, Grad2, Grad3])))
fig2.set_yticks(np.logspace(gm, gM, num=gM-gm+1))
fig2.set_xlabel('Number of SGD training steps', fontdict=font2)
fig2.set_ylabel('Norm of the gradient', fontdict=font2)
fig3.semilogx(xx, Param1, color = 'gray')
fig3.semilogx(xx, Param2, color = 'darkorange')
fig3.semilogx(xx, Param3, color = 'teal')
fig3.set_xlabel('Number of SGD training steps', fontdict=font2)
fig3.set_ylabel(r'$\mathcal{L}^{2}\!$'+'-distance to limit point', fontdict=font2)
plt.savefig("Opt_SNN_SGD_10_100_1000_clipping_biases_train.pdf", bbox_inches="tight")
plt.show()