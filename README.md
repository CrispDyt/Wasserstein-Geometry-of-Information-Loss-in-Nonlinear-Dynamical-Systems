# Wasserstein Geometry of Information Loss in Nonlinear Dynamical Systems

Code relating to the PRSA Paper 

Duan Y, Zhang Z, Guo Y. Wasserstein Geometry of Information Loss in Nonlinear Dynamical Systems[J]. arXiv preprint arXiv:2601.22814, 2026. https://arxiv.org/abs/2601.22814

## Dataset Description

nycmeas.data: measles outbreak data from New York City between 1928 and 1964  

Stady_Data.mat_Final_Result_17_9_5_4.996.mat: double pendulum trajectories 

## Usage for Main Figures and Plots.

To generate results of CCM and plot Figure 1 and 3, please run the notebook Figure1_CCM_Experiments.ipynb. 

To generate results of Figure 2, please run the notebook Figure2.ipynb.

To generate results of Figure 5, please run the notebook Figure5_Double.ipynb and Figure5_Measles.ipynb. 

To generate results of Figure 6 (EDMD), please run the notebook Figure6_Downstream_EDMD.ipynb. 

To generate results of Figure 7, please run the notebook Figure7_Downstream_DIM.ipynb and plot Figure 7 please run Figure7_Plot_DIM.ipynb. 

## Usage for Tables 

Since our estimator is based on k-NN, which can be parallelly implemented, we use the jit function from the numba package. 

To generate results of Table 1 and 2

'python Table_1_Rossler_Parallel.py'

To generate results of Table 4 

'python Table_2_Rossler_sensitive_analysis.py'







