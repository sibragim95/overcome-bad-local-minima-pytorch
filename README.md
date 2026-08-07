# Overcome Bad Local Minima (PyTorch Implementation)

This repository provides a **PyTorch** re-implementation of key numerical simulations from the research paper:

> **"Convergence to good non-optimal critical points in the training of neural networks: Gradient descent optimization with one random initialization overcomes all bad non-global local minima with high probability"**  
> *Authors:* Shokhrukh Ibragimov, Arnulf Jentzen, Adrian Riekert  
> *Preprint:* [arXiv:2212.13111](https://arxiv.org/abs/2212.13111)

---

## 🔬 About This Implementation

While the [original repository](https://github.com/deeplearningmethods/overcome-bad-local-minima) contains all numerical experiments implemented in **TensorFlow**, this repository refactors a selected subset of the **core simulations (Section 4)** into **PyTorch**. 

### Key Highlights:
* **Strict Mathematical Fidelity:** Prioritized exact 1-to-1 mathematical translation over standard PyTorch object-oriented design: Utilized functional closures to perfectly mimic TensorFlow 1.x's static computational graphs, ensuring that untrainable parameters remain strictly frozen while target variables are updated.
* **True Monte Carlo Dynamics:** Carefully restructured the training loops to evaluate fresh Monte Carlo batches dynamically at every single (Adam) SGD step, guaranteeing true population risk minimization rather than static empirical risk minimization.
* **High-Performance & Portable:** Engineered with device-agnostic execution for instant deployment. The code automatically leverages high-end hardware (e.g., NVIDIA GPUs) to reduce 10,000-step simulation times from hours down to minutes, while retaining graceful fallbacks for local CPU execution.

---

## 🛠️ Repository Structure

```text
├── Codes/
│   ├── Clipping_SNN_all_biases_PyTorch.py           # Training all biases of a shallow NN with clipping activation
│   ├── ReLU_Adam_DNN_4Layer_Xavier_PyTorch.py       # Training a deep 4-layer ReLU NN with Adam optimizer and Xavier init
│   ├── ReLU_DNN_all_params_Random_Normal_PyTorch.py # Training all parameters of a deep ReLU NN with random normal init
│   └── ReLU_SNN_inner_bias_PyTorch.py               # Training only inner biases of a shallow ReLU NN
├── .gitignore                                       # Python ignore rules
├── LICENSE                                          # MIT License
└── README.md                                        # Repository documentation
