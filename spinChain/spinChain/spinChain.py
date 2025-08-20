import numpy as np
import math
from itertools import combinations, permutations
import numba
from numba import jit
import time
import matplotlib.pyplot as plt
#import libspinChain as ls


@jit(nopython = True, cache = True)
def rootsOfUnity(numSites, k):
    return k**(1/numSites) * np.exp(2 * np.pi * 1j / numSites) ** np.arange(numSites)

@jit(nopython = True, cache = True)
def createInitialSolution(positionState, numSites):
    # TODO: Rewrite this in terms of the roots of unity.
    roots = np.roots(np.array([1] + [0]*(numSites-1) + [(-1)**len(positionState)], dtype=np.complex128)) # This should probably be done in numpy, but it doesn't matter
    # Return the numSites roots of unity multiplied by (-1)**(numUp + 1)
    # roots = (-1)**(len(positionState) + 1) * np.exp(2 * np.pi * 1j / numSites) ** np.arange(numSites)
    # roots = ((-1)**len(positionState))**(1/numSites) * np.exp(2 * np.pi * 1j / numSites) ** np.arange(numSites)
    return roots[positionState]

@jit(nopython = True, cache = False)
def BetheEquation(initial, i, delta):
    numUp = len(initial)
    term = 1
    # print(initial)
    for j in range(numUp):
        # NOTE: This calculation could be easily vectorized or done in numba, and maybe just mathematically simplified
        term *= (-1) * ((1 + initial[i]*initial[j] - 2*delta*initial[i])/(1 + initial[i]*initial[j] - 2*delta*initial[j]))
    # print(-term)
    return -term

@jit(nopython = True, cache = True)
def iterateBetheAnsatz(initial, delta, numSites, unityRoots):
    numUp = len(initial)
    w = initial.copy()
    # w = np.zeros(numUp, dtype=np.complex128)
    # unityRoots = rootsOfUnity(numSites, 1)
    for i in range(numUp):
        # roots = np.roots(np.array([1] + [0]*(numSites-1) + [-BetheEquation(initial,i,delta)], dtype=np.complex128))
        # roots = rootsOfUnity(numSites, BetheEquation(initial, i, delta))
        roots = BetheEquation(initial, i, delta)**(1/numSites) * unityRoots
        # print(f'Roots = {roots - initial[i]}')
        # print(f'Initial[i] = {initial[i]}')
        w[i] = roots[np.argmin( np.abs( roots-initial[i] ) )]
    return w

@jit(nopython = True, cache = True)
def computeBetheRoots(allConfigurations, numSites, delta, thresholdSq = 1e-28, maxIt = 1000):
    '''
    numUp is the number of up spins
    numSites is the number of sites
    delta is the coupling constant
    numUpdates is the number of updates for the Bethe ansatz process
    '''

    allBetheRoots = np.ones(allConfigurations.shape, dtype=np.complex128)
    numIt = np.empty(allConfigurations.shape[0])
    # Use map to return a list rather than a tuple.  This makes the indexing work better.
    unityRoots = rootsOfUnity(numSites, 1)
    for i, positionState in enumerate(allConfigurations):
        # TODO: Rewrite this in terms of the roots of unity only
        allBetheRoots[i] = createInitialSolution(positionState, numSites)
        # current = unityRoots[positionState]
        # print(positionState, current)
        previous = allBetheRoots.copy() + 100
        # print(f'Initial value = {initial}')
        # print(positionState, initial)
        # Numsteps should be replaced w/ a while loop and some test for convergence
        it = 0
        while np.sum(np.abs(allBetheRoots[i]-previous)**2) > thresholdSq and it < maxIt:
            previous = allBetheRoots[i].copy()
            allBetheRoots[i] = iterateBetheAnsatz(allBetheRoots[i], delta, numSites, unityRoots)
            it += 1
            if it >= maxIt:
                print(f'{i} failed to converge')
            # if i == 431:
                # print(np.sum(np.abs(allBetheRoots[i]-previous)**2))
        # print(it)
        numIt[i] = it
        # systemCounter += 1
        # print(systemCounter, stepCounter)
        # roots.append(current)
        # break
    return allBetheRoots, numIt

def precomputeAllInversions(n):
    # Return a list of all of the pairs for which sigma[i] > sigma[j], given i > j
    sigma = np.array(list(permutations(range(n))))
    inversions = []
    inversionPointer = np.zeros(sigma.shape[0]+1, dtype=int)
    
    for permIndex in range(sigma.shape[0]):
        for i in range(n):
            for j in range(i+1,n):
                if sigma[permIndex, i] > sigma[permIndex, j]:
                    inversions.append([sigma[permIndex, i], sigma[permIndex, j]])
        inversionPointer[permIndex+1] = len(inversions)
    # If n is 1 then there will be no inversions and so the inversion array will have dimensions that cause numba to choke.  So, put in dummy data
    if n == 1:
        inversions = [[-1]]
    return sigma, np.array(inversions), inversionPointer
    # return inversions, np.unique(np.array(inversions), axis=0, return_counts=True)

@jit(nopython = True, cache = True)
def gaudinLikeMatrix(betheRoots, delta, numSites):
    numerator = -2*delta * (1 + betheRoots**2 - 2*delta*betheRoots)
    denominator = (1 + np.outer(betheRoots, betheRoots) - 2*delta*betheRoots[:,np.newaxis]) * (1 + np.outer(betheRoots, betheRoots) - 2*delta*betheRoots)
    mat = numerator/denominator
    np.fill_diagonal(mat, -np.sum(mat,1) + np.diag(mat) + numSites/betheRoots)
    return mat

@jit(nopython = True, cache = True)
def computeAmplitude(betheRoots, invList, delta):
    amplitude = np.complex128(1)
    # TODO: Maybe replace this with a call to np.prod()
    for i in range(invList.shape[0]):
        amplitude *= -(1 + betheRoots[invList[i,0]]*betheRoots[invList[i,1]] - 2*delta*betheRoots[invList[i,0]])/(1 + betheRoots[invList[i,0]]*betheRoots[invList[i,1]] - 2*delta*betheRoots[invList[i,1]])
    return amplitude

@jit(nopython = True, cache = True)
def intPower(base, exponent):
    result = np.complex128(1)
    while exponent > 0:
        if exponent % 2 == 1:
            result *= base
        base *= base
        exponent //= 2
    return result

@jit(nopython = True, cache = True)
def computeBasisTransformMatrices(allBetheRoots, allConfigurations, sigma, inversions, inversionPointer, delta, numSites):
    # TODO: We can compute the actual eigenvectors by finding the degenerate pairs of betheVectors and replacing them with their sum and their difference.  More
    # effectively, we have pairs of the form v, w such that v = w*.  So, we can replace them with Re(v) and Im(v) to get eigenvectors in the "preferred basis".
    # If we do this then we will have a matrix V such that V\dagger V = identity and H = V\dagger Lambda V.  This means that we don't actually need to explicitly
    # compute both basis transform matrices.

    # Eqn's 21 (u = configToEnergy) and 55 (l = energyToConfig) in URSA23
    energyToConfig = np.zeros((allBetheRoots.shape[0], allConfigurations.shape[0]), dtype=np.complex128) # This is a square matrix for converting from the position basis to the eigen-basis
    configToEnergy = np.zeros((allConfigurations.shape[0], allBetheRoots.shape[0]), dtype=np.complex128)
    
    # Computational complexity: O[(L choose N)^2 * N! * N * Log(N)]
    for energyIndex, betheRoots in enumerate(allBetheRoots): # L choose N
        gaudinDet = np.linalg.det(gaudinLikeMatrix(betheRoots, delta, numSites))
        betheProduct = np.prod(betheRoots)
        for permIndex in range(sigma.shape[0]): # O(N!)
            if inversionPointer[permIndex+1] > inversionPointer[permIndex]:
                # print(betheRoots)
                # print(inversions[inversionPointer[permIndex]:inversionPointer[permIndex+1]])
                amplitude = computeAmplitude(betheRoots, inversions[inversionPointer[permIndex]:inversionPointer[permIndex+1]], delta)
            else:
                amplitude = np.complex128(1)
        
            # print(f"EnergyIndex: {energyIndex} permIndex: {permIndex} amplitude= ", amplitude)

            for positionIndex, positionState in enumerate(allConfigurations): # O(L choose N)
                # Replace the np.prod with an unwrapped for loop for a small speed-up (10 s -> 9.5 s)
                # product = np.prod(betheRoots[sigma[permIndex]]**positionState)
                product = np.complex128(1)
                # Almost all of the runtime is used computing this product.  Because position state is an integer we can unroll that as well       
                for i in range(len(betheRoots)): # O(N)
                    # # product *= betheRoots[sigma[permIndex, i]] ** positionState[i] # This takes about 9.5s for N=5, L=11          
                    # for j in range(positionState[i]): # This takes about 1.75 seconds for N=5, L=11
                    #     product *= betheRoots[sigma[permIndex, i]]
                    # Replace with an O(log(n)) integer power function
                    product *= intPower(betheRoots[sigma[permIndex, i]], positionState[i]) # O(Log(N)), This takes about 1.5 s for N=5, L=11
                # print(product)
                ###### TODO: This is the correct behavior

                energyToConfig[energyIndex, positionIndex] += amplitude * product 
                # if energyIndex == 0 and positionIndex == 0:
                #     print('XXXXXXXXXXXXXXXXXXXXXXXXX ',product)
                #     print('e2c[0,0]= ', energyToConfig[0,0])
                configToEnergy[positionIndex, energyIndex] += 1/ ( gaudinDet * betheProduct * amplitude * product )

                ###### TODO: This is debugging
                # energyToConfig[energyIndex, positionIndex] += product
                # configToEnergy[positionIndex, energyIndex] += 1/ ( gaudinDet * betheProduct * amplitude * product )                
    # energyToConfig, configToEnergy = normalizeMatrices(energyToConfig, configToEnergy)
    return energyToConfig, configToEnergy

def computeEigenBasis(configToEnergy, energyList):
    # The columns of configToEnergy are the basis transform vectors: c2e[:,i] is the ith vector
    eigenBasis = np.zeros(configToEnergy.shape, dtype=float)


def getSpectrum(numUp, numSites, delta):
    print(delta)
    allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    allBetheRoots, numIt = computeBetheRoots(allConfigurations, numSites, delta)
    energyList = computeEnergy(allBetheRoots, delta)
    return energyList, numIt

# def dryRunMatrixCalculation(allBetheRoots, allConfigurations, delta, numSites, numUp):
def dryRunMatrixCalculation(numUp, numSites, delta):

    # InitialCondition is a list of the locations of up spins
    # betheRoots is a list of length numSites choose numUp of lists of length numUp
    # allConfigurations is all of the possible ways of distributing those numUp states
    # delta is the coupling strength
    # numSites is the number of sites
    # numUp is the number of up spins

    sAll = time.time()
    allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    print(f'allConfigurations = {time.time()-sAll}')
    
    s = time.time()
    allBetheRoots, numIt = computeBetheRoots(allConfigurations, numSites, delta)
    energyList = computeEnergy(allBetheRoots, delta)
    print(f'allBetheRoots = {time.time()-s}')
    # allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    s = time.time()
    sigma, inversions, inversionPointer = precomputeAllInversions(numUp)
    print(f'precomputeAllInversions = {time.time()-s}')
    
    s=time.time()
    energyToConfig, configToEnergy = computeBasisTransformMatrices(allBetheRoots, allConfigurations, sigma, inversions, inversionPointer, delta, numSites)
    print(f'computeBasisTransformMatrices = {time.time()-s}')
    print(f'TotalTime = {time.time() - sAll}')
    return energyToConfig, configToEnergy, allBetheRoots, energyList, allConfigurations, sigma, inversions, inversionPointer


# Code here to interface with pybind11 code
# Ideally we call the code as
# energyToConfig, configToEnergyTrans = computeBasisTransform((array) allBetheRoots, (array) allConfigurations, (vector) allGaudinDets, (array) sigma, complex delta, long numSites)

def pybind11BasisTransform(numUp, numSites, delta):
    
    sAll = time.time()
    allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    print(f'allConfigurations = {time.time()-sAll}')
    
    s = time.time()
    allBetheRoots, numIt = computeBetheRoots(allConfigurations, numSites, delta)
    energyList = computeEnergy(allBetheRoots, delta)
    print(f'allBetheRoots = {time.time()-s}')
    # allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    s = time.time()
    sigma = np.array(list(permutations(range(numUp))))
    print(f'computePermutations = {time.time()-s}')
    
    s=time.time()
    # Precompute all gaudin determinants
    allGaudinDets = np.array([np.linalg.det(gaudinLikeMatrix(betheRoots, delta, numSites)) for betheRoots in allBetheRoots])
    print(f'computeGaudinDets = {time.time()-s}')

    s=time.time()
    energyToConfig, configToEnergyTrans = ls.computeBasisTransform(allBetheRoots, allConfigurations, allGaudinDets, sigma, delta, numSites)
    # computeBasisTransformMatrices(allBetheRoots, allConfigurations, sigma, inversions, inversionPointer, delta, numSites)
    print(f'computeBasisTransformMatrices = {time.time()-s}')
    print(f'TotalTime = {time.time() - sAll}')
    return np.array(energyToConfig), np.array(configToEnergyTrans), allBetheRoots, energyList, allConfigurations, sigma

def pyCUDABasisTransform(numUp, numSites, delta):
        
    sAll = time.time()
    allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    print(f'allConfigurations = {time.time()-sAll}')
    
    s = time.time()
    allBetheRoots, numIt = computeBetheRoots(allConfigurations, numSites, delta)
    nBasis = allBetheRoots.shape[0]
    energyList = computeEnergy(allBetheRoots, delta)
    print(f'allBetheRoots = {time.time()-s}')
    # allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    s = time.time()
    sigma = np.array(list(permutations(range(numUp))))
    print(f'computePermutations = {time.time()-s}')
    
    s=time.time()
    # Precompute all gaudin determinants
    allGaudinDets = np.array([np.linalg.det(gaudinLikeMatrix(betheRoots, delta, numSites)) for betheRoots in allBetheRoots])
    print(f'computeGaudinDets = {time.time()-s}')

    s=time.time()
    e2c, c2e = ls.gpuComputeBasisTransform(allBetheRoots.flatten(), allConfigurations.flatten(), allGaudinDets, sigma.flatten(), delta)
    # energyToConfig, configToEnergyTrans = ls.computeBasisTransform(allBetheRoots, allConfigurations, allGaudinDets, sigma, delta, numSites)
    # # computeBasisTransformMatrices(allBetheRoots, allConfigurations, sigma, inversions, inversionPointer, delta, numSites)
    print(f'[GPU] computeBasisTransformMatrices = {time.time()-s}')
    print(f'TotalTime = {time.time() - sAll}')
    return np.array(e2c).reshape(nBasis, nBasis), np.array(c2e).reshape(nBasis, nBasis), allBetheRoots, energyList, allConfigurations, sigma, allGaudinDets

def pyCUDABasisTransformDegen(numUp, numSites, delta):
        
    sAll = time.time()
    allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    print(f'allConfigurations = {time.time()-sAll}')
    
    s = time.time()
    allBetheRoots, numIt = computeBetheRoots(allConfigurations, numSites, delta)
    nBasis = allBetheRoots.shape[0]
    print(f'allBetheRoots = {time.time()-s}')

    s = time.time()
    energyList = computeEnergy(allBetheRoots, delta)
    degenList = degenEnergies(energyList)
    indepList = np.array([d[0] for d in degenList])
    if np.max([len(d) for d in degenList]) > 2:
        raise ValueError("pyCUDABasisTransformDegen only works for degenericies up to 2")
    print(f'nBasis = {nBasis}, nIndep = {indepList.shape[0]}')
    print(f'degenList = {time.time() -s}')

    # allConfigurations = np.array(list(combinations(range(numSites), numUp)))
    s = time.time()
    sigma = np.array(list(permutations(range(numUp))))
    print(f'computePermutations = {time.time()-s}')
    
    # s=time.time()
    # # Precompute all gaudin determinants
    # allGaudinDets = np.array([np.linalg.det(gaudinLikeMatrix(betheRoots, delta, numSites)) for betheRoots in allBetheRoots])
    # print(f'computeGaudinDets = {time.time()-s}')

    s=time.time()
    e2cDegen = ls.gpuComputeBasisTransformDegen(allBetheRoots[indepList,:].flatten(), allConfigurations.flatten(), sigma.flatten(), delta, numUp)
    # return np.array(e2cDegen)
    e2cDegen = np.array(e2cDegen).reshape(indepList.shape[0], nBasis)
    # energyToConfig, configToEnergyTrans = ls.computeBasisTransform(allBetheRoots, allConfigurations, allGaudinDets, sigma, delta, numSites)
    # # computeBasisTransformMatrices(allBetheRoots, allConfigurations, sigma, inversions, inversionPointer, delta, numSites)
    print(f'[GPU] computeBasisTransformMatrices = {time.time()-s}')

    s=time.time()
    e2cStd, c2eStd = constructFullBasis(e2cDegen, degenList)
    print(f'constructFullBasis = {time.time()-s}')

    print(f'TotalTime = {time.time() - sAll}')
    return e2cStd, c2eStd, allBetheRoots, energyList, allConfigurations, sigma


def computeEnergy(betheRoots, delta):
    energyList = np.sum(betheRoots + 1/betheRoots - 2*delta, 1)
    if np.max(np.abs(np.imag(energyList))) > np.finfo(float).eps*1e3:
        raise ValueError(f"energyList should be real-valued, max imag part is {np.max(np.abs(np.imag(energyList)))}")

    return np.real(energyList)

def evolvePositionState(initialPositionState, configToEnergy, energyToConfig, energyList, time):
    betheState = np.dot(configToEnergy, initialPositionState)
    evolvedBetheState = betheState * np.exp(-1j * time * energyList)
    evolvedPositionState = np.dot(energyToConfig, evolvedBetheState)
    return evolvedPositionState

def normalizeMatrices(energyToConfig, configToEnergy):
    # NOTE: One needs to use the same normalization to avoid introducing precision errors so that e2c@c2e = identity
    e2cNorm = np.linalg.norm(energyToConfig, axis=1)
    c2eNorm = np.linalg.norm(configToEnergy, axis=0)
    # TODO: This transpose stuff should be handled more elegantly
    return energyToConfig/e2cNorm, (configToEnergy.T * e2cNorm).T
    # return normalizeMatrix(energyToConfig), normalizeMatrix(configToEnergy.T).T

def normalizeMatrix(m):
    return m / np.linalg.norm(m, axis=0)

def degenEnergies(energyList, tol=1e-10):
    order = np.argsort(energyList)
    edges = np.hstack([0, 1+np.where(np.diff(energyList[order]) > tol)[0], order.shape[0]])
    return [order[edges[i]:edges[i+1]] for i in range(edges.shape[0]-1)]

def constructFullBasis(e2cDegen, degenList):
    e2cStd = np.empty([e2cDegen.shape[1]]*2, dtype=complex)
    for i, degen in enumerate(degenList):
        if len(degen) == 2:
            e2cStd[degen,:] = e2cDegen[i,:].real, e2cDegen[i,:].imag
            e2cStd[degen,:] /= np.linalg.norm(e2cStd[degen,:], axis=1).reshape(2,1)
        else:
            e2cStd[degen[0],:] = e2cDegen[i,:]/np.linalg.norm(e2cDegen[i,:])

    c2eStd = e2cStd.conj().T
    return e2cStd, c2eStd

def makeStandardForm(e2c, energyList, tol=1e-10):
    dList = degenEnergies(energyList, tol=tol)
    # e2cStd = normalizeMatrix(e2c)
    e2cStd = e2c.copy()
    for degen in dList:
        if len(degen) == 2:
            e2cStd[degen[0],:] = (e2c[degen[0]] + e2c[degen[1]])/2
            e2cStd[degen[0],:] /= np.linalg.norm(e2cStd[degen[0],:])
            e2cStd[degen[1],:] = (e2c[degen[0]] - e2c[degen[1]])/2
            e2cStd[degen[1],:] /= np.linalg.norm(e2cStd[degen[1],:])
        if len(degen) == 1:
            e2cStd[degen[0]] /= np.linalg.norm(e2cStd[degen[0],:])
    # e2cStd = normalizeMatrix(e2c)
    c2eStd = e2cStd.conj().T.copy()
    return e2cStd, c2eStd

def reconstructHamiltonian(e2c, c2e, energyList):
    return c2e@np.diag(energyList)@e2c

def hamiltonian(numUp, numSites, delta):
    # Define spin matrices
    
    Id = np.array([[1, 0],
                   [0, 1]], dtype=complex)

    Sx = np.array([[0, 1],
                   [1, 0]], dtype=complex)

    Sy = np.array([[0,   1j],
                   [-1j, 0]], dtype=complex)

    Sz = np.array([[1,  0],
                   [0, -1]], dtype=complex)

    # local operator
    h = (1/2)*np.kron(Sy, Sy) + (1/2)*np.kron(Sx, Sx) + (delta/2)*(np.kron(Sz, Sz) - np.kron(Id,Id))

    # sum of local operators
    H = np.kron(h, np.eye(2**(numSites-2)))
    
    for k in range(1, numSites-2):
        H += np.kron(np.eye(2**k), np.kron(h, np.eye(2**(numSites-2-k))))
    H += np.kron(np.eye(2**(numSites-2)), h)

    Hreturn = H.copy()

    # local operator at the ends
    ends = (1/2)*np.kron(Sy, np.kron(np.eye(2**(numSites-2)), Sy)) 
    ends += (1/2)*np.kron(Sx, np.kron(np.eye(2**(numSites-2)), Sx))
    ends += (delta/2)*(np.kron(Sz, np.kron(np.eye(2**(numSites-2)), Sz)) - np.eye(2**numSites))

    #Hamiltonian for the ring with all particles
    H += ends

    #Indexes with not N up-spins
    Nindex =[]
    for n in range(2**numSites):
        occ = [int(b) for b in bin(n)[2:]] #occupation configuration
        if sum(occ) != numSites-numUp:
            Nindex.append(n)

    # Hamiltonian sector with only N up-spins
    HN = np.delete(H,Nindex, axis=0)
    HN = np.delete(HN, Nindex, axis=1)

    return HN, H, Hreturn

def benchmark(min=3, max=10):
    allTimes = []
    numElements = []
    nList = np.arange(min,max,2)
    for n in nList:
        start = time.time()
        sol = computeBetheRoots(n, 2*n-1, 0.03)
        allTimes.append(time.time()-start)
        numElements.append(sol.shape[0])
        print(n, allTimes[-1], allTimes[-1]/numElements[-1])
    return nList, np.array(allTimes), np.array(numElements)




HN, H, Hreturn = hamiltonian(4,9,0.07)

e2c, c2e, allBetheRoots, energyList, allConfigurations, sigma, inversions, inversionPointer = dryRunMatrixCalculation(4,9,0.07)
reconstructedH = reconstructHamiltonian(e2c,c2e,energyList)

print('Difference between reconstructed hamiltonian and original hamiltonian: \n', np.real(reconstructedH-HN))
plt.imshow(np.real(reconstructedH-HN), cmap='viridis',vmin=0,vmax=1)
plt.colorbar(label='Value')
plt.show()

# if __name__ == '__main__':
#     # TODO: Seems to fail when N=1
#     # c2e, e2c, allBetheRoots, allConfigurations, sigma, inversions, inversionPointer = dryRunMatrixCalculation(2,5,.03)
#     c2e, e2c, allBetheRoots, allConfigurations, sigma, inversions, inversionPointer = dryRunMatrixCalculation(4,9,.03)