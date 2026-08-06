# Overcome Bad Local Minima (PyTorch Implementation)

This repository provides a **PyTorch** re-implementation of key numerical simulations from the research paper:

> **"Convergence to good non-optimal critical points in the training of neural networks: Gradient descent optimization with one random initialization overcomes all bad non-global local minima with high probability"**  
> *Authors:* Shokhrukh Ibragimov, Arnulf Jentzen, Adrian Riekert  
> *Preprint:* [arXiv:2212.13111](https://arxiv.org/abs/2212.13111)

---

## 🔬 About This Implementation

While the [original repository](https://github.com/deeplearningmethods/overcome-bad-local-minima) contains all numerical experiments implemented in **TensorFlow**, this repository refactors a selected subset of the **core simulations (Section 4)** into **PyTorch**. 

### Key Highlights:
* **Selective Refactoring:** Rather than duplicating near-identical code across all 10 numerical setups, this repository focuses on re-implementing the key representative experiments that showcase gradient descent convergence dynamics.
* **Modern PyTorch Design:** Refactored static TensorFlow 1.x computational graphs into clean, modular, and dynamic PyTorch workflows.
* **Reproducibility:** Optimized parameter handling for quick execution and visualization of loss trajectories and critical point avoidance.

---

## 🛠️ Repository Structure

```text
├── codes/
│   ├── simulation_core.py      # Main training & optimization script in PyTorch
│   └── utils.py                # Loss functions, initialization, and plotting routines
├── .gitignore                  # Python ignore rules
├── LICENSE                     # MIT License
└── README.md                   # Repository documentation
