"""
Author: Justin Angra
Advisor: Richard J. Funstahl
Last Updated: October 10, 2014

Usage: ME_Euler.py [-h] -m KINETIC_FILE -v POTENTIAL_FILE -s N_STEPS
"""

import sys
import argparse
from time import clock  # timeing
from math import factorial
from utils import bernoulli
import numpy as np
import matplotlib.pyplot as plt  # plotting
plt.ion()  # allows plots via IPython

def myComm(op1, op2):
    """Returns the commutator relation for given input.
    For commuting values function returns 0.

    USAGE:
        myComm(op1, op2)

    INPUT:
        op1    - matrix of the same dimentions as op2

        op2    - matrix of the same dimentions as op1
    OUTPUT:
             - matrix representing the commutator relation between op1 and op2
    """
    return np.dot(op1, op2)-np.dot(op2, op1)


def adjointOp(op1, op2, order):
    """Returns the adjoint relation for the given input.

    USAGE:
        adjointOp(op1, op2, order)

    INPUT:
        op1   - matrix of the same dimentions as op2

        op2   - matrix of the same dimentions as op1

        order - integer representing degree of adjoint operator

    OUTPUT:
          - matrix of size mat_size populated with the adjoint operator relation
         up to order
    """
    # base case first order adjoint operator
    if order == 0:
        return op1

    # higher order adjoint operator
    else:
        return myComm(op2, adjointOp(op1, op2, order-1))


def funct_dOmeg(H0, T, Omeg, order):
    """
    Return ODEs

    USAGE:
        funct_dOmeg(H0, T, Omeg, order)

    INPUT:
        H0   - matrix representing initial value of the Hamiltonian (T + V)

        T   - matrix representing kinetic values

        Omeg - matrix representing Omega values

        order - integer representing to what order to take the summation

    OUTPUT:
        dOmeg solutions from given Omeg
    """
    Hs = funct_Hs(H0, Omeg, order)

    eta = myComm(T, Hs)
    dOmeg = np.copy(eta)
    for i in xrange(1, order):  # length of order
        dOmeg += bernoulli(i) / factorial(i) * adjointOp(eta, Omeg, i)
    return dOmeg


def funct_Hs(H0, Omeg, order):
    """
    Return Hs (Hamiltonian)

    USAGE:
        funct_Hs(H0, Omeg, order)

    INPUT:
        H0   - matrix representing initial value of the Hamiltonian (T + V)

        Omeg - matrix representing Omega values

        order - integer representing to what order to take the summation

    OUTPUT:
        Hs values from given Omeg
    """
    Hs = np.copy(H0)  # create a copy of H0 on Hs
    for i in xrange(1, order):
        Hs += ((1. / factorial(i))) * adjointOp(H0, Omeg, i)
    return Hs


def eulerMethod(Omeg0, h, t00, H0, T, order, n_steps):
    """
    Return ODEs solutions

    USAGE:
        eulerMethod(Omeg0, h, t00, H0, T, order, n_steps)

    INPUT:

        Omeg0 - matrix zeros initial Omeg values

        h - step size

        t00 - initial time mesh to evolve over

        H0   - matrix representing initial value of the Hamiltonian (T + V)

        T   - matrix representing kinetic values

        order - integer representing to what order to take the summation

        n_steps - number of euler method steps

    OUTPUT:
        Hs values from given Omeg
    """    
    t = [t00]  # initial time
    soln = []  # Solutions
    soln.append(Omeg0)

    g = np.copy(Omeg0)
    
    # Euler Method Steps
    for step in xrange(n_steps):       
        s = g + h * 1.0 * funct_dOmeg(H0, T, g, order)
        g = np.copy(s)
        t00 += h
        t.append(t00)
        soln.append(s)
    return soln, t


def plotData(soln, mesh):
    """Plots the time evolution of de-coupled DEs with respect
    to mesh (time-grid)
    
    USAGE:
        plotData(soln, mesh)

    INPUT:

        soln - array of array of numerical solutions
        
        mesh - time grid to plot over
    """
    # Plot diagonal and off diagonal elements of Hs from soln
    plt.figure()
    for i in xrange(len(soln)):
        plt.plot(mesh, soln[i], label='H%s' % i)  # plot with corresponding label
    plt.axhline(y=0, color='black', linestyle='dotted')  # y=0 line
    plt.xlabel('lambda = 1/s ^(1/4)')  # x-axis label
    plt.ylabel('Energy')  # y-axis label
    plt.title('SRG Evolution via ME')  # title
    #plt.legend(loc=(.6, .5), ncol=2, shadow=True)  # location of legend
    plt.show()
    return


def main():   
    # parser to get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--matrix', dest='matrix_file', required=True)
    parser.add_argument('-v', '--potential', dest='vector_file', required=True)
    parser.add_argument('-s', '--steps', type=int, dest='n_steps', required=True)
    globals().update(vars(parser.parse_args()))

    # Read Matrix T and V
    try:
        T = np.loadtxt(matrix_file, dtype=np.float)  # Read T matrix file
        V = np.loadtxt(vector_file, dtype=np.float)  # Read V matrix file
    except Exception as e:
        print e
        sys.exit(2)

    # Chceck if both matrices are of the same size
    assert(len(T) == len(V)), "Matrices do not have same dimensions" 
    
     # Check that V is symmetric
    np.testing.assert_equal(V, np.transpose(V), err_msg = "\nPotential Matrix is not symmetric\n")
        
    # Calculate Matrix Size
    mat_size = len(T)

    # define initial hamiltonian in terms of kinetic and potenial
    H0 = T + V

    order = 5  # truncate both series to the 5th order

    Omeg0 = np.zeros((mat_size, mat_size), dtype=np.float)  # create initial value matrix

    sMesh = np.linspace(0, 1, 1000)  # define time mesh
    h = sMesh[1] - sMesh[0]  # Step size
    t00 = sMesh[0]  # initial time

    start = clock()
    soln, t = eulerMethod(Omeg0, h, t00, H0, T, order, n_steps)  # Execute Euler Method to solve ODEs
    print "Total run time solving ODEs: {0} seconds".format(clock() - start)

    # Substite ODEs solution on first equation to get Hs
    Hs = []
    for omeg in soln:
        Hs.append(funct_Hs(H0, omeg, order))

    lambMesh = map(lambda x: 1. / x**(1. / 4), t)  # for plotting to infinity

    # Convert Matrix solutions to array
    arrTmp = Hs[0].ravel()
    for i in xrange(1, n_steps+1):
        arrTmp = np.row_stack((arrTmp, Hs[i].ravel()))
    arrSoln = []
    for i in xrange(0, mat_size**2):
        arrSoln.append(arrTmp[:, i].tolist())

    # Plot Hs
    plotData(arrSoln, lambMesh)
    
    return

if __name__ == "__main__":
    start = clock()
    main()
    print "Total run time: {0} seconds".format(clock() - start)
