# Wasserstein Geometry of Information Loss in Nonlinear Dynamical Systems

Code relating to the PRSA Paper 

Duan Y, Zhang Z, Guo Y. Wasserstein Geometry of Information Loss in Nonlinear Dynamical Systems[J]. arXiv preprint arXiv:2601.22814, 2026. https://arxiv.org/abs/2601.22814

## Dataset Description

nycmeas.data: measles outbreak data from New York City between 1928 and 1964  

Stady_Data.mat_Final_Result_17_9_5_4.996.mat: double pendulum trajectories 

## Usage for Main Figures and Plots.

- Figures 1 and 3 (CCM): run Figure1_CCM_Experiments.ipynb.
- Figure 2: run Figure2.ipynb.
- Figure 5: run Figure5_Double.ipynb and Figure5_Measles.ipynb.
- Figure 6 (EDMD): run Figure6_Downstream_EDMD.ipynb.
- Figure 7: run Figure7_Downstream_DIM.ipynb to generate the results, and then run Figure7_Plot_DIM.ipynb to reproduce the figure.

## Usage for Tables 

Since our estimator is based on k-NN, which can be parallelly implemented, we use the jit function from the numba package. 

To generate results of Table 1 and 2

```bash
python Table_1_Rossler_Parallel.py
```

To generate results of Table 4 

```bash
python Table_2_Rossler_sensitive_analysis.py
```







