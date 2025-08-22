print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re
from scipy import constants
from scipy import integrate
from math import floor, log10
print('imported')

# Constants
timestep = 0.005 # lj
nevery = 100 # how often data is dumped
indextime = timestep * nevery # time between each data point in picoseconds

sigma = 22.5E-9 # m
epsilon = 2.25E-20 # J
mass = 2.12E-22 # kg
F = (epsilon/sigma) # N
T_LJ = 1.0 #18.4
Temp = T_LJ*epsilon/constants.Boltzmann # K
tau = (sigma)*np.sqrt(mass/epsilon) # s

r_0_prime = 2

print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')


def integrand1UU(r2, r1, k_r_prime):
    return r1**2 * r2**2 * np.exp(-(U_r_u(r1, k_r_prime)+U_r_u(r2, k_r_prime))/T_LJ) 

def integrand2UU(r2, r1, k_r_prime):
    r = r1 + r2
    return r* r1**2 * r2**2 * np.exp(-(U_r_u(r1, k_r_prime)+U_r_u(r2, k_r_prime))/T_LJ)

def Q_UU(k_r_prime):
    rmax = r_0_prime*(1-1e-9)
    result = integrate.nquad(
        integrand1UU,
        [[0, rmax], [0, rmax]],
        args=(k_r_prime,)
    )
    return result[0]

def Expected_r_UU(k_r_prime):
    rmax = r_0_prime*(1-1e-9)
    x = integrate.nquad(
        integrand2UU,
        [[0, rmax], [0, rmax]],
        args=(k_r_prime,)
    )
    result = x[0]/Q_UU(k_r_prime)
    print(f'<r_UU> = {result}, force = {F}')
    return result

def U_r_u(r, k_r_prime):
    if r >= r_0_prime:
        return np.inf
    return (-0.5*k_r_prime*r_0_prime**2)*np.log(1-r**2/r_0_prime**2)

def main():
    k = np.linspace(1, 100, 1000)
    # t = np.linspace(0.01,10,10)
    # q = np.linspace(0.01,2,1000)
    R_mean = np.array([Expected_r_UU(i) for i in k])
    # Z = np.array([[Zr_bond(i,j)/integrate.quad(Zr_bond, 0, 2, args=(j)) for i in q] for j in t])

    Delta_R_mean = np.abs(R_mean-1-0.1)

    plt.figure(1)
    plt.xlabel('k')
    plt.ylabel('Delta')
    plt.title('|<r> - 1.1|')
    plt.plot(k, Delta_R_mean)

    print(f'Best K_r_prime = {k[np.argmin(Delta_R_mean)]} based on mean.')

    plt.show()

main()