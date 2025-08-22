# Function of script:
# 1) If VaryForce global parameter is True, create histograms that show unfolded distribution shift with force applied.
# 2) If VaryForce global parameter is False, create histograms that show unfolded distribution shift with varying spring constant values.

# Data in output/ForceSeedUU must reflect these changes, and data folder titling must be as follows:
# 1) VaryForce: 
#       Use run_fibrin.sh to generate a folder titled ForceSeedUU. 
#       Its contents must have data stored in folders of the format: Force{force}_Seed{seed}, force must be a decimal.
# 2) Not VaryForce: 
#       Use run_fibrinff.sh to generate a folder titled K_rt_SeedUU. 
#       Its contents have data stored in folders of the format: Seed{seed}_k_r{k_r}_k_theta{k_theta}, all must be decimals.

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
F_LJ = (epsilon/sigma) # N
T_LJ = 1.0 #18.4
Temp = T_LJ*epsilon/constants.Boltzmann # K
tau = (sigma)*np.sqrt(mass/epsilon) # s

k_theta = 7.785
k_r = 43.87
k_r_prime = 3
r_0 = 1
r_0_prime = 2

print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F_LJ} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')
def readfile():
    # Change directory to the project folder
    print(f'current dir: {os.getcwd()}')
    currentdir = os.getcwd()
    path = os.path.join(currentdir, r'output/ForceSeedUU')
    os.chdir(path)
    print(os.getcwd())

    folder_lst = sorted(os.listdir())
    rows = []

    folder_num = len(folder_lst)
    counter = 1
    
    for folder in folder_lst:
        # https://docs.python.org/3/library/re.html
        # https://docs.python.org/3/howto/regex.html#regex-howto
        # output/k_r${k_r}_k_theta${k_theta}
        seed_match = re.search(r'Seed(\d+)?', folder)
        r_match = re.search(r'Force(\d+(?:\.\d+)?)', folder)

        if not (r_match and seed_match):
            continue
        
        seed = int(seed_match.group(1))
        force = float(r_match.group(1))

        folder_path = os.path.join(os.getcwd(), folder)
        os.chdir(folder_path)

        btype_lst = [[],[]]
        blen_lst = [[],[]]
        with open('bondlengths.dump') as f:
            isItem = False
            counter2 = 0
            for line in f:
                if 'ENTRIES c_btype c_bondlen' in line:
                    isItem = True
                    counter2 = 0
                    continue
                if not isItem:
                    continue
                if counter2 == 2:
                    isItem = False
                    counter2 = 0
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    btype_lst[counter2].append(int(parts[0]))
                    blen_lst[counter2].append(float(parts[1]))
                counter2 += 1
                
        bangle_lst = []
        with open('angles.dump') as f:
            isItem = False
            for line in f:
                if 'ITEM: ENTRIES c_myAngle' in line:
                    isItem = True
                    continue
                if isItem:
                    try:
                        bangle_lst.append(float(line.strip()))
                    except ValueError:
                        pass
                    isItem = False
                
        
        mlen_arr = np.array(blen_lst, dtype=np.float64)[0, :] + np.array(blen_lst, dtype=np.float64)[1, :]
        blen_arr = np.array(blen_lst, dtype=float) 
        row = {
            "seed": seed,

            "force": force,

            "types": np.array(btype_lst, dtype=int),

            "bond lengths": blen_arr,       
            "monomer lengths": mlen_arr,       

            "angle_deg": np.array(bangle_lst, dtype=float),
        }
        
        rows.append(row)
        os.chdir('..')
        print(f'{counter}/{folder_num} folders read')
        counter += 1

    df = pd.DataFrame(rows)
    df.sort_values(by=['force'], inplace=True)

    max_len = df['monomer lengths'].apply(len).max()

    if not df[df['monomer lengths'].apply(len) < max_len].empty:
        print('##########################################################')
        print('Simulation did not run to completion for the following: ')
        print(df[df['monomer lengths'].apply(len) < max_len])
        print('##########################################################')

    df = df[df['monomer lengths'].apply(len) == max_len]

    df1 = df.groupby(['force']).agg({
        'monomer lengths':       lambda s: np.concatenate(s.to_numpy()),
        'angle_deg':     lambda s: np.concatenate(s.to_numpy()),
        'types':         lambda s: np.concatenate(s.to_numpy()),
        'bond lengths':  lambda s: np.concatenate([np.asarray(x, float) for x in s], axis=1),
    }).reset_index()
    return df1

def integrand1UU(r, F):
    def integral(r1, r2, F):
        x = r1**2 * np.exp(-(U_r_u(r1))/T_LJ) * Q_rot(r1, r2, F)
        return x
    return r**2 * np.exp(-(U_r_u(r))) * integrate.quad(integral, 0, r_0_prime, args=(r, F))[0]

def integrand2UU(r1, r2, F):
    return r1**2 * r2**2 * np.exp(-(U_r_u(r1)+U_r_u(r2))/T_LJ) * Q_rot(r1, r2, F)

def Q_UU(F):
    rmax = r_0_prime*(1-1e-9)
    result = integrate.nquad(
        integrand2UU,
        [[0, rmax], [0, rmax]],
        args=(F,)
    )
    return result[0]

def U_r_u(r):
    if r >= r_0_prime:
        return np.inf
    return (-0.5*k_r_prime*r_0_prime**2)*np.log(1-r**2/r_0_prime**2)

def Q_rot(r1, r2, F):
    alpha =(r1+r2)*F/(T_LJ)

    if F == 0:
        x = 4*np.pi
    else:
        x = 4*np.pi*sinhc(alpha)

    return x
    
def sinhc(x):
    if np.isclose(x, 0.0):
        return 1.0 + x**2/6.0
    elif x > 50:  # large argument
        return 0.5 * np.exp(x - np.log(x))  # = (e^x / 2)/x
    else:
        return np.sinh(x)/x

    

def main():
    data_df = readfile()
    data_df.sort_values(by=['force'], inplace=True)
    
    # monomer_df = stats(data_df)

    os.chdir('..')
    os.chdir('..')
    path = os.path.join(os.getcwd(), r'figures')
    os.chdir(path)
    print(os.getcwd())

    plt.rcParams.update({
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    print()
    print('Using data:')
    print(data_df)
    print()

    intlim = np.array([0,r_0_prime*(1 - 1E-2)])
    fig, ax = plt.subplots()
    ax.set_box_aspect(2/3)
    x = np.linspace(*intlim,200)

    plt.figure(1)
    for j in range(0,3):
        i = 1*j
        F = data_df['force'].iloc[i]
        r = np.array([integrand1UU(l, F) for l in x])
        Z = Q_UU(F)
        print(Z)
        plt.plot(x, r/Z, label=f"force = {F} lj theory")  
        plt.hist(data_df['bond lengths'].iloc[i][0,:], bins=500, density=True, alpha=0.5, label=f"force = {F} lj")

    plt.xlabel('Bond 1 length (lj)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('UU_bond1_length_histogram_force.eps', bbox_inches='tight')
    plt.savefig('UU_bond1_length_histogram_force.png', bbox_inches='tight')

    plt.figure(2)
    for j in range(0,3):
        print(f'j {j}')
        i = 1*j
        F = data_df['force'].iloc[i]
        print(f'force {F}, counter {i}')
        r = np.array([integrand1UU(l, F) for l in x])
        Z = Q_UU(F)
        print(f'Z 2: {Z}')
        plt.plot(x, r/Z, label=f"force = {F} lj theory")  
        plt.hist(data_df['bond lengths'].iloc[i][1,:], bins=500, density=True, alpha=0.5, label=f"force = {F} lj")
    plt.xlabel('Bond 2 length (lj)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('UU_bond2_length_histogram_force.eps', bbox_inches='tight')
    plt.savefig('UU_bond2_length_histogram_force.png', bbox_inches='tight')

    plt.show()

main()