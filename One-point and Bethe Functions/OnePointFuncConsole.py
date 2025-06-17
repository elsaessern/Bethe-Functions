import numpy as np
import scipy.special as sp
import random
import matplotlib
from matplotlib import pyplot as plt
from itertools import permutations
import os       # to access environment vars 
import sys      # for command line args 
import time     # for timing stuff 

def M(z, i, j, d):
    return d*( 1 + z[i]*z[i] - d*z[i] )/((1 + z[i]*z[j] - d*z[j])*(1 + z[i]*z[j] - d*z[i]))
	
def Cmat(z, d, L):
    n = len(z)
    diag =0
    mat = [[ - M(z, i, j, d) for i in range(n)] for j in range(n) ]
    for i in range(n):
        diag = sum(mat[i]) +M(z, i, i, d)
        mat[i][i] = -diag + (L/z[i])
    return(mat)

def Coeff(y, z, d, L):
    n = len(z)
    terms = [ z[i]**(y[i]+1) for i in range(n)] 
    coeff = np.prod(terms)
    coeff = coeff*np.linalg.det(Cmat(z, d, L))
    coeff = 1/coeff
    return coeff

def inversions(s):
    n = len(s)
    inv = []
    for i in range(n):
        for j in range(i+1, n):
            if s[i] > s[j]:
                inv.append((s[i],s[j]))
    return inv

def A(z, s, d):
    n = len(z)
    inv = inversions(s)
    terms = [ -(1 + z[i[0]]*z[i[1]] - d*z[i[0]])/(1 + z[i[0]]*z[i[1]] - d*z[i[1]]) for i in inv]
    return np.prod(terms)

def eigenfun(x, z, d):
    n = len(z)
    per = list(permutations(range(n)))
    final_term=0
    term = 0
    for s in per:
        term = A(z, s,d)
        for i in range(n):
            term = term*(z[s[i]]**(x[i]))
        final_term = final_term + term
    return final_term

def BE(z,i,d):
    n = len(z)
    term=1
    for j in range(n):
        term= term*(-1)*((1 + z[i]*z[j] - d*z[i])/(1 + z[i]*z[j] - d*z[j]))
    return -term

def seq_update(z,d,L):
    n = len(z)
    w = []
    for i in range(n):
        pol = [1]
        pol.extend([0]*(L-1))
        pol.append(- BE(z, i, d))
        roots = np.roots(pol)
        roots = [x-z[i] for x in roots]
        w.append(min(roots, key=np.abs) + z[i])
    return w

def BE_sol_initial(I,L):
    n = len(I)
    pol = [1]
    pol.extend([0]*(L-1))
    pol.append((-1)**n)
    roots = np.roots(pol)
    sol =[]
    for i in I:
        sol.append(roots[i])
    return sol

def tuple_combinations(N,L):
    def helper(current_tuple, index):
        
        if index == N:
            result.append(current_tuple)
            return
        if index ==0:
            for i in range(L):
                helper(current_tuple + (i,), index+1)
        else:
            for i in range(current_tuple[-1]+1, L):
                helper(current_tuple + (i,), index+1) 
    result = []
    helper(tuple(), 0)
    return result

def BE_all_sol(N,L,d,updates):
    all_tuples = tuple_combinations(N,L)
    sol = []
    for I in all_tuples:
        initial = BE_sol_initial(I,L)
        for k in range(updates):
            initial = seq_update(initial, d, L)
        sol.append(initial)
    return sol

def energy(z,d):
    return sum([i +1/i - 2*d for i in z])

def partition(sol,d,b):
    return sum([np.exp(-b*energy(z,d)) for z in sol])

def sum_energy(sol, d, b):
    return sum([ energy(z,d)*np.exp(-b*energy(z,d))for z in sol] )

def subCv2(j,config):
    sC = []
    N= len(config)
    for n in range(N):
        if j in config[n]:
            sC.append(n)
    return(sC)

def sum_terms(y, aS, aC, d, L):
    return [[ Coeff(y, z, d, L)*eigenfun(x, z, d) for x in aC] for z in aS]

def prob_configv2(x_index, t, aS, sT,d):
    N = len(aS)
    terms = [sT[k][x_index]*np.exp(-1j*energy(aS[k],d)*t) for k in range(N)]
    return(np.abs(sum(terms))**2)

def prob_one_v2(x_one , t,aS, aC, sT,d):
    subConfig = subCv2(x_one, aC)
    terms = [prob_configv2(k, t, aS, sT,d) for k in subConfig]
    return(sum(terms))

# argument checks 
if len(sys.argv) < 3:
    print("ERROR: Not enough arguments. Requires L, N, and delta as arguments from the command line.", file=sys.stderr)
    exit()

# reading in arguments
N = int(sys.argv[1])
L = int(sys.argv[2])
d = float(sys.argv[3])

# some integrity checks (for N and initial values) 
if N < 0 or N >= L:
    print("ERROR: N cannot be outside the range [0, L).")
    exit()
if len(sys.argv) < N + 3:
    print("ERROR: Not enough arguments. Requires initial conditions after L, N, and delta from the command line.", file=sys.stderr)
    exit()

IC_one = [int(sys.argv[i]) for i in range(4, N+4)]
for initCond in IC_one:
    if initCond < 0 or initCond > L:
        print("ERROR: Invalid initial condition. Every initial value should be in [0, L) (got {} for a chain of length {}).".format(initCond, L), file=sys.stderr)
        exit()

# hardcoded argument (number of updates to numerical method 
# for solving the Bethe Ansatz equations)
acc = 10000

# COMPUTATION START 
startTime = time.time()

# PLOT THINGS
all_C = tuple_combinations(N,L)
ord_sol = BE_all_sol(N,L, d, acc)
all_sol = []
for z in ord_sol:
    all_sol.extend(list(permutations(z)))

all_T = sum_terms(IC_one, all_sol, all_C, d, L)

x = np.arange(0, L-1, 1)
t = np.arange(0, L, 0.1)


X, T = np.meshgrid(x, t)
Z = np.array([[prob_one_v2(x1, t1 , all_sol, all_C, all_T,d)  for x1 in x] for t1 in t])
#Z = prob_one(Step_IC, X, T, all_sol, d, N, L)

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

print("t=0 probabilities: ", Z[0,:])

#cmap yellow to blue: plt.cm.cividis

surf = ax.plot_surface(X, T, Z, cmap = 'Oranges', vmin=0, vmax=1)

# Set axes label
plt.title("One Point Probability Function (delta = {})".format(d))
ax.set_xlabel('x, position (arb.u.)', labelpad=20)
ax.set_ylabel('t, time (arb.u.)', labelpad=20)
# ax.set_zlabel('P(X(t) = x), probability of particle at position x at time t', labelpad=20)



ax.view_init(25, 35)

plotFileName = 'n{} l{} d{} {}'.format(N, L, d, IC_one)
if 'SLURM_JOB_ID' in os.environ:
    plotFileName = plotFileName + " (job {})".format(os.environ['SLURM_JOB_ID'])
plotFileName = plotFileName + ".png"

plt.colorbar(surf, shrink=0.5, aspect=8)


plt.savefig("Plots and output files/"+plotFileName)
# plt.show()

ax.view_init(90,0)
ProjectedPlotFileName = "2D " + plotFileName
plt.savefig("Plots and output files/"+ProjectedPlotFileName)

# COMPUTATION END 
print("This computation took {} seconds.".format(time.time() - startTime))
