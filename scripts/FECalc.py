print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re
from scipy import constants
from scipy import integrate
from scipy import special
from math import floor, log10
print('imported')

# Point of script:
# One/two bond unfolded fractions
# Mean time one/two unfolded + std/se
# Free energy + standard error

# Constants
timestep = 0.00001 # lj
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
k_r = 21.935*2
k_r_prime = 3
r_0_prime = 2
r_0 = 1

print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F_LJ} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')

def readfile():
    # Change directory to the project folder
    print(f'current dir: {os.getcwd()}')
    currentdir = os.getcwd()
    path = os.path.join(currentdir, r'output/ForceSeedFF')
    # path = r'\\wsl.localhost\Ubuntu\home\jacob\projects\LammpsCode\output\CORRuu'
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
                
        # bangle_lst = []
        # with open('angles.dump') as f:
        #     isItem = False
        #     for line in f:
        #         if 'ITEM: ENTRIES c_myAngle' in line:
        #             isItem = True
        #             continue
        #         if isItem:
        #             try:
        #                 bangle_lst.append(float(line.strip()))
        #             except ValueError:
        #                 pass
        #             isItem = False
                

        mlen_arr = np.array(blen_lst, dtype=np.float64)[0, :] + np.array(blen_lst, dtype=np.float64)[1, :]
        blen_arr = np.array(blen_lst, dtype=float) 
        row = {
            "force": force,
            "seed": seed,

            "types": np.array(btype_lst, dtype=int),

            "bond lengths": blen_arr,       
            "monomer lengths": mlen_arr,       

            # "angle_deg": np.array(bangle_lst, dtype=float),
        }
        
        rows.append(row)
        os.chdir('..')
        print(f'{counter}/{folder_num} folders read')
        counter += 1

    df = pd.DataFrame(rows)

    df1 = df.groupby(['force']).agg({
        'monomer lengths':       lambda s: np.concatenate(s.to_numpy()),
        # 'angle_deg':     lambda s: np.concatenate(s.to_numpy()),
        #'types':         lambda s: np.concatenate(s.to_numpy()),
        'bond lengths':  lambda s: np.concatenate([np.asarray(x, float) for x in s], axis=1),
        'monomer lengths': lambda s: np.concatenate([np.asarray(x, float) for x in s])
    }).reset_index()
    return df1



def round_sig(x, sig=2):
    # https://stackoverflow.com/a/3413529
    return round(x, sig-int(floor(log10(abs(x))))-1)

def integrand1FF(r2, r1, F):
    return r1**2 * r2**2 * np.exp(-(U_r_f(r1)+U_r_f(r2))/T_LJ) * Q_rot(r1, r2, F)

def integrand2FF(r2, r1, F):
    r = r1 + r2
    return r * r1**2 * r2**2 * np.exp(-(U_r_f(r1)+U_r_f(r2))/T_LJ) * Q_rot(r1, r2, F)

def integrand1UF(r2, r1, F):
    return r1**2 * r2**2 * np.exp(-(U_r_f(r1)+U_r_u(r2))/T_LJ) * Q_rot(r1, r2, F)

def integrand2UF(r2, r1, F):
    r = r1 + r2
    return r * r1**2 * r2**2 * np.exp(-(U_r_f(r1)+U_r_u(r2))/T_LJ) * Q_rot(r1, r2, F)

def integrand1UU(r2, r1, F):
    return r1**2 * r2**2 * np.exp(-(U_r_u(r1)+U_r_u(r2))/T_LJ) * Q_rot(r1, r2, F)

def integrand2UU(r2, r1, F):
    r = r1 + r2
    return r* r1**2 * r2**2 * np.exp(-(U_r_u(r1)+U_r_u(r2))/T_LJ) * Q_rot(r1, r2, F)

def Q_UU(F):
    rmax = r_0_prime*(1-1e-9)
    result = integrate.nquad(
        integrand1UU,
        [[0, rmax], [0, rmax]],
        args=(F,)
    )
    return result[0]
def Q_UF(F):
    rmax = r_0_prime*(1-1e-9)
    rmax2 = r_0 * 5
    result = integrate.nquad(
        integrand1UF,
        [[0, rmax], [0, rmax2]],
        args=(F,)
    )
    return result[0]

def Q_FF(F):
    rmax = r_0 * 5
    result = integrate.nquad(
        integrand1FF,
        [[0, rmax], [0, rmax]],
        args=(F,)
    )
    return result[0]

def Expected_r_UU(F):
    rmax = r_0_prime*(1-1e-9)
    result = integrate.nquad(
        integrand2UU,
        [[0, rmax], [0, rmax]],
        args=(F,)
    )
    return result[0]/Q_UU(F)

def Expected_r_UF(F):
    rmax = r_0_prime*(1-1e-9)
    rmax2 = r_0 * 20
    result = integrate.nquad(
        integrand2UF,
        [[0, rmax], [0, rmax2]],
        args=(F,)
    )
    return result[0]/Q_UF(F)

def Expected_r_FF(F):
    rmax = r_0 * 5
    result = integrate.nquad(
        integrand2FF,
        [[0, rmax], [0, rmax]],   # ranges for r2, r1
        args=(F,)
    )
    return result[0]/Q_FF(F)

# def Z_bend():
#     return integrate.quad(Z_bend_integrand, 0, np.pi)[0]

# def Z_bend_integrand(theta):
#     return np.sin(theta)*np.exp(-E_bend(theta)/T_LJ)

# def E_bend(theta):
#     return 0.5*k_theta*(theta-np.pi)**2

def U_r_u(r):
    if r >= r_0_prime:
        return np.inf
    return (-0.5*k_r_prime)*np.log(1-r**2/r_0_prime**2)

def U_r_f(r):
    return 0.5*k_r*(r-r_0)**2

def Q_rot(r1, r2, F):
    alpha =(r1+r2)*F/(T_LJ)

    if F == 0:
        x = 4*np.pi
    else:
        x = 4*np.pi*sinhc(alpha)

    return x
    
def sinhc(x):
    if np.isclose(x, 0.0):
        return 1.0 + x**2/6.0  # series expansion
    else:
        return np.sinh(x)/x
    
def stats(data_df):
    rt_pairs = data_df['force'].drop_duplicates().values
    

    std_mlen = []
    sem_std_mlen = []
    mean_mlen = []
    sem_mlen = []

    mean_blen = []

    for force in rt_pairs:
        try:
            std_mlen.append(np.std(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))
            sem_std_mlen.append((np.std(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['force'] == force]['monomer lengths'].iloc[0])))
            
            mean_mlen.append(np.mean(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))
            sem_mlen.append((np.mean(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['force'] == force]['monomer lengths'].iloc[0])-2))
            mean_blen.append(np.mean(np.array(data_df[data_df['force'] == force]['bond lengths'].iloc[0]), axis=1))

        except ValueError as e:
            print(f"Error calculating std for f={force}: {e}")
            std_mlen.append(np.nan)
            sem_std_mlen.append(np.nan)

            mean_mlen.append(np.nan)
            mean_mlen.append(np.nan)


    monomer_df = pd.DataFrame({
        "force": data_df['force'],

        "monomer length mean": mean_mlen,
        "monomer length sem": sem_mlen,
        "monomer length std": std_mlen,
        "monomer length std sem": sem_std_mlen,

        "blen mean": mean_blen,

        "lengths": data_df['monomer lengths'].values
        # "angles": data_df['angle_deg'].values,
    })
    
    return monomer_df 


def main():
    data_df = readfile().sort_values('force')
    monomer_df = stats(data_df).sort_values('force')
    F1 = np.linspace(0.1,50,50)
    x_FF = np.array([Expected_r_FF(i) for i in F1])
    # x_UF = np.array([Expected_r_UF(i) for i in F1])
    # F2 = np.linspace(0.1,150,75)
    # x_UU = np.array([Expected_r_UU(i) for i in F2])

    os.chdir('..')
    os.chdir('..')
    currentdir = os.getcwd()
    path = os.path.join(currentdir, r'figures')

    plt.figure(1)
    plt.title('FF Monomer Force-Extension Curve (Harmonic)')
    # plt.ylim(-3,55)
    # plt.xlim(1.9,4.8)
    plt.xlabel('Monomer Length (lj)')
    plt.ylabel('Force (lj)')
    # plt.plot(x_UU, F2, label='UU')
    # plt.plot(x_UF, F1, label='FU')
    plt.plot(x_FF, F1, label='FF Theory')
    # plt.scatter(monomer_df['monomer length mean'], monomer_df['force'])
    plt.errorbar(monomer_df['monomer length mean'], monomer_df['force'], yerr=monomer_df['monomer length sem'], fmt='.', )
    # plt.plot(monomer_df['monomer length mean'], monomer_df['force'])
    plt.legend()
    
    plt.savefig('monomer_force_extension_FF.eps', bbox_inches='tight')

    plt.figure(2)
    plt.plot(np.array(monomer_df['blen mean'].tolist())[:, 0],monomer_df['force'], marker='o')
    plt.plot(np.array(monomer_df['blen mean'].tolist())[:, 1],monomer_df['force'], marker='o')
    plt.xlabel('mean bond length (lj)')
    plt.ylabel('force (lj)')
    plt.show()
main()