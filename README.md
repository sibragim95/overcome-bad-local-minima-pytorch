# Overcome Bad Local Minima (PyTorch Implementation)

This repository provides a **PyTorch** re-implementation of key numerical simulations from the research paper:

> **"Convergence to good non-optimal critical points in the training of neural networks: Gradient descent optimization with one random initialization overcomes all bad non-global local minima with high probability"**  
> *Authors:* Shokhrukh Ibragimov, Arnulf Jentzen, Adrian Riekert  
> *Preprint:* [arXiv:2212.13111](https://arxiv.org/abs/2212.13111)

---

## 🔬 About This Implementation

While the [original repository](https://github.com/deeplearningmethods/overcome-bad-local-minima) contains all numerical experiments implemented in **TensorFlow**, this repository refactors a selected subset of the **core simulations (Section 4)** into **PyTorch**. 

### Key Highlights:
* **Strict Mathematical Fidelity:** Prioritized exact 1-to-1 mathematical translation over standard PyTorch object-oriented design. We utilized functional closures to perfectly mimic TensorFlow 1.x's static computational graphs, ensuring that untrainable parameters remain strictly frozen while target variables are updated.
* **True Monte Carlo Dynamics:** Carefully restructured the training loops to evaluate fresh Monte Carlo batches dynamically at every single SGD step, guaranteeing true population risk minimization rather than static empirical risk minimization.
* **High-Performance & Portable:** Engineered with device-agnostic execution for instant deployment. The code automatically leverages high-end hardware (e.g., NVIDIA GPUs) to reduce 10,000-step simulation times from hours down to minutes, while retaining graceful fallbacks for local CPU execution.

---

## 🛠️ Repository Structure

```text
├── codes/
│  ├── Clipping_SNN_all_biases_PyTorch.py     # Training only biases of shallow NN (SNN) with clipping activation
│  ├── ReLU_SGD_DNN_4Layer_Xavier.py          # Training all parameters of deep ReLU NN with Xavier initialization
│  └── ReLU_SNN_inner_bias.py                 # Training only inner biases of shallow ReLU NN
├── .gitignore           # Python ignore rules
├── LICENSE              # MIT License
└── README.md            # Repository documentation
