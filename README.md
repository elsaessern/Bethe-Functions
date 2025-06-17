# Bethe-Functions

What is this?
----
This is an open source GitHub repository containing code for the functions introduced in (link arxiv paper here) related to the Bethe Ansatz for the Heisenberg-Ising (XXZ) Spin-1/2 chain. You will find code which contains definitions for the functions themselves along with code to generate plots for observables of the model (only the one-point function for now). Additionally, there is code written to run on GPUs in high performance computing clusters.

Why?
---
* Provide source code for the plots and numerical evidence for the conjectures found in (link arxiv paper here)
* Advance research and simulation in the fields relating to the Bethe Ansatz and XXZ spin chain by providing open source access to relevant code

How do I get started? 
---
* For basic implementation and functions found in (link arxiv paper here), the folder 'One-point and Bethe Functions' contains the file 'DetFormula.ipynb' which provides a basic implementation of the Bethe Functions and numerical verification of the inverse function for the coordinate Bethe Ansatz. The 'OnePointFuncConsole.py' file is Python code which generates plots for the One-point function.
* Plots of the one-point function and related output files can be found in the 'Plots and output files' folder
* The 'spinChain-main' folder contains the GPU implementation of code for the Bethe Functions and a more extensive implementation for numerically verifying the inverse function for the coordinate Bethe Ansatz.  
