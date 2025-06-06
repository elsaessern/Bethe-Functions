import numpy as np
import scipy.special as sp
import random
import matplotlib
from matplotlib import pyplot as plt
from itertools import permutations
import os       # to access environment vars 
import sys      # for command line args 
import time     # for timing stuff
import json
import sympy

# Entries of the Guadin-like matrix
def M(z, i, j, d):
    return d*( 1 + z[i]*z[i] - d*z[i] )/((1 + z[i]*z[j] - d*z[j])*(1 + z[i]*z[j] - d*z[i]))

# Gaudin-like matrix	
def Cmat(z, d, L):
    n = len(z)
    diag =0
    mat = [[ - M(z, i, j, d) for i in range(n)] for j in range(n) ]
    for i in range(n):
        diag = sum(mat[i]) +M(z, i, i, d)
        mat[i][i] = -diag + (L/z[i])
    return(mat)

# Coefficients for to write the configuration basis as a linear combination of Bethe vectors
def Coeff(y, z, d, L):
    n = len(z)
    terms = [ z[i]**(y[i]+1) for i in range(n)] 
    coeff = np.prod(terms)
    coeff = coeff*np.linalg.det(Cmat(z, d, L))
    coeff = 1/coeff
    return coeff

# All the inversion of a permuation s
def inversions(s):
    n = len(s)
    inv = []
    for i in range(n):
        for j in range(i+1, n):
            if s[i] > s[j]:
                inv.append((s[i],s[j]))
    return inv

# Amplitude coefficients for the Bethe vectors
def A(z, s, d):
    n = len(z)
    inv = inversions(s)
    terms = [ -(1 + z[i[0]]*z[i[1]] - d*z[i[0]])/(1 + z[i[0]]*z[i[1]] - d*z[i[1]]) for i in inv]
    return np.prod(terms)

# Bethe vectors
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

#all configurations in that include the index j
def subCv2(j,config):
    sC = []
    N= len(config)
    for n in range(N):
        if j in config[n]:
            sC.append(n)
    return(sC)

#all configurations that have a gap (i.e. no particles) in [j+1, k) and a particle at j
def gap_subC(j, k, Config):
    sC = []
    gap = True
    for con in Config:
        gap = True
        if j in con:
            for x in range(j+1,k):
                gap = gap and not x in con
        else:
            gap= False
        if gap:
            sC.append(con)
    return sC

#all indexes configurations that have a gap (i.e. no particles) in [j+1, k) and a particle at j
def gap_subCv2(j, k, Config):
    sC = []
    N= len(Config)
    gap = True
    for n in range(N):
        gap = True
        if j in Config[n]:
            for x in range(j+1, k):
                gap = gap and not x in Config[n]
        else:
            gap = False
        if gap:
            sC.append(n)
    return sC

#partitions all of the configurations Config based on their last element, as long as the last element are in the range [j,k-1]
def gap_subCv3(j, k, Config):
    sC = [[] for x in range(j, k)]
    N = len(Config)
    for n in range(N):
        last = Config[n][-1]
        if last >= j and last < k:
            sC[last-j].append(n)
    return sC


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

# probability that there is a gap (i.e. no particle) in [j+1, k) with a particle at j
def prob_gap(j,k, t, aS, aC, sT, d):
    subConfig = gap_subCv2(j, k, aC)
    terms = [prob_configv2(c, t, aS, sT, d) for c in subConfig]
    return(sum(terms))

# probability vector for the last particle to be at position x in [j, k-1]
def prob_gap_v2(j, k, t, aS, aC, sT, d):
    parConfig = gap_subCv3(j, k, aC)
    probs =[]
    for n in range(k-j):
        terms = [prob_configv2(c, t, aS, sT, d) for c in parConfig[n]]
        probs.append(sum(terms))
    return(probs)

# argument checks 
#if len(sys.argv) < 3:
#    print("ERROR: Not enough arguments. Requires L, N, and delta as arguments from the command line.", file=sys.stderr)
#    exit()

# reading in arguments
N = 2
L = 11
d = 0

# some integrity checks (for N and initial values) 
if N < 0 or N >= L:
    print("ERROR: N cannot be outside the range [0, L).")
    exit()
#if len(sys.argv) < N + 3:
#    print("ERROR: Not enough arguments. Requires initial conditions after L, N, and delta from the command line.", file=sys.stderr)
#    exit()

IC_one = [4,5]
for initCond in IC_one:
    if initCond < 0 or initCond > L:
        print("ERROR: Invalid initial condition. Every initial value should be in [0, L) (got {} for a chain of length {}).".format(initCond, L), file=sys.stderr)
        exit()

# hardcoded argument (number of updates to numerical method 
# for solving the Bethe Ansatz equations)
# this may be causing a bottle neck?
acc = 2

# COMPUTATION START 
startTime = time.time()

# PLOT THINGS
all_C = tuple_combinations(N,L)
startTime2 = time.time()
ord_sol = BE_all_sol(N,L, d, acc)
startTime3 = time.time()
all_sol = []
for z in ord_sol:
    all_sol.extend(list(permutations(z)))
startTime4 = time.time()
all_T = sum_terms(IC_one, all_sol, all_C, d, L)
startTime5 = time.time()
x = np.arange(0, L, 1)
##t = np.arange(0, 8, 0.1)


##X, T = np.meshgrid(x, t)

#one-point probability function
##Z = np.array([[prob_one_v2(x1, t1 , all_sol, all_C, all_T,d)  for x1 in x] for t1 in t])

#gap probability function at time t = L/8
##Z2 = np.array([prob_gap(x1, L, L/8,all_sol, all_C, all_T, d ) for x1 in x])

#gap probability function at time t = L/8 (faster?)
Z3 = prob_gap_v2(0,L, L/8, all_sol, all_C, all_T, d)
startTime6 = time.time()
#print(Z2)
#print(Z3)

##Z = prob_one(Step_IC, X, T, all_sol, d, N, L)


# convert np arrays to lists since json doesnt support np arrays
##Zl = Z2.tolist()

Zl = Z3

#open file and write it
with open('gap_probs.json', 'w+') as f:
    json.dump(Zl, f)
    print("File written successfully")
f.close()


'''
fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')


surf = ax.plot_surface(X, T, Z, cmap = plt.cm.cividis)

# Set axes label
plt.title("One Point Probability Function (delta = {})".format(d))
ax.set_xlabel('x, position (arb.u.)', labelpad=20)
ax.set_ylabel('t, time (arb.u.)', labelpad=20)
# ax.set_zlabel('P(X(t) = x), probability of particle at position x at time t', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

ax.view_init(90, 0)

plotFileName = 'n{} l{} d{} {}'.format(N, L, d, IC_one)
if 'SLURM_JOB_ID' in os.environ:
    plotFileName = plotFileName + " (job {})".format(os.environ['SLURM_JOB_ID'])
plotFileName = plotFileName + ".png"

plt.savefig(plotFileName)
plt.show()
'''


# COMPUTATION END 
print("This computation took {} seconds.".format([time.time() - startTime6,startTime6-startTime5,startTime5-startTime4, startTime4-startTime3, startTime3-startTime2, startTime2-startTime ] ))
