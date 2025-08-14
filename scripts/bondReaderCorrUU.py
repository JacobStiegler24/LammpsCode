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

# Point of script:
# One/two bond unfolded fractions
# Mean time one/two unfolded + std/se
# Free energy + standard error

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

k_theta = 7.785

print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')
def readfile():
    # Change directory to the project folder
    print(os.getcwd())
    os.chdir('..')
    print(os.getcwd())
    path = os.path.join(os.getcwd(), r'\output\CORRuu')
    path = r'\\wsl.localhost\Ubuntu\home\jacob\projects\LammpsCode\output\CORRuu'
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
        r_match = re.search(r'k_r_prime(\d+(?:\.\d+)?)', folder)

        if not (r_match and seed_match):
            continue
        
        seed = int(seed_match.group(1))
        k_r_lmmps = float(r_match.group(1))

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

            "K_r phys": k_r_lmmps,

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
    df1 = df.groupby(['K_r phys']).agg({
        'monomer lengths':       lambda s: np.concatenate(s.to_numpy()),
        'angle_deg':     lambda s: np.concatenate(s.to_numpy()),
        'types':         lambda s: np.concatenate(s.to_numpy()),
        'bond lengths':  lambda s: np.concatenate([np.asarray(x, float) for x in s], axis=1),
    }).reset_index()
    return df1

def round_sig(x, sig=2):
    # https://stackoverflow.com/a/3413529
    return round(x, sig-int(floor(log10(abs(x))))-1)


def Zr_bond(r, k_r):
    integrand = r**2*((1-(r**2)/4)**(k_r*2))
    return integrand

def ExpectedRIntegral_bond(r, k_r):
    return Zr_bond(r,k_r)*r

def ExpectedRSqrIntegral_bond(r, k_r):
    return Zr_bond(r,k_r)*r**2

def ExpectedR(k_r):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k_r,))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k_r,))
    expectedR_new = invZ*integral*2
    return expectedR_new

def ExpectedRSqr(k_r):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k_r,))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral1, err = integrate.quad(ExpectedRSqrIntegral_bond, 0, 2, args=(k_r,))
    integral2, err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k_r,))
    expectedR_sqr_new = invZ*integral1*2 + 2*(invZ*integral2)**2

    return expectedR_sqr_new

def VarR(k_r):
    
    expectedR = ExpectedR(k_r)
    expectedRSqr = ExpectedRSqr(k_r)

    var = expectedRSqr-expectedR**2
    return var

def stats(data_df):
    rt_pairs = data_df['K_r phys'].drop_duplicates().values
    

    std_mlen = []
    sem_std_mlen = []
    mean_mlen = []
    sem_mlen = []

    mean_blen = []

    for k_r in rt_pairs:
        try:
            std_mlen.append(np.std(data_df[data_df['K_r phys'] == k_r]['monomer lengths'].iloc[0]))
            sem_std_mlen.append((np.std(data_df[data_df['K_r phys'] == k_r]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['K_r phys'] == k_r]['monomer lengths'].iloc[0])))
            
            mean_mlen.append(np.mean(data_df[data_df['K_r phys'] == k_r]['monomer lengths'].iloc[0]))
            sem_mlen.append((np.mean(data_df[data_df['K_r phys'] == k_r]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['K_r phys'] == k_r]['monomer lengths'].iloc[0])))
            mean_blen.append(np.max(np.array(data_df[data_df['K_r phys'] == k_r]['bond lengths'].iloc[0]), axis=1))

        except ValueError as e:
            print(f"Error calculating std for k_r={k_r}: {e}")
            std_mlen.append(np.nan)
            sem_std_mlen.append(np.nan)

            mean_mlen.append(np.nan)
            mean_mlen.append(np.nan)


    monomer_df = pd.DataFrame({
        "K_r": data_df['K_r phys'],
        "K_theta": data_df['K_theta phys'],

        "monomer length mean": mean_mlen,
        "monomer length sem": sem_mlen,
        "monomer length std": std_mlen,
        "monomer length std sem": sem_std_mlen,

        "blen mean": mean_blen,

        "lengths": data_df['monomer lengths'].values,
        "angles": data_df['angle_deg'].values,
    })
    
    return monomer_df 

def main():
    data_df = readfile()
    data_df.sort_values(by=['K_r phys'], inplace=True)
    
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

    x = np.linspace(0.005, 90, 300)
    p = np.sqrt(np.array([VarR(i) for i in x]))
    l = np.array([ExpectedR(i) for i in x])
    y = p/l

    spring_const_vals = data_df[['K_r phys']].drop_duplicates()
    ri1 = -1
    ri2 = 0


    plt.figure(1)
    fig, ax = plt.subplots()
    ax.set_box_aspect(2/3)
    x = np.linspace(0,2,200)

    r = np.array([Zr_bond(i, spring_const_vals['K_r phys'].iloc[ri1]) for i in x])
    Z, err = integrate.quad(Zr_bond, 0, 2, args=spring_const_vals['K_r phys'].iloc[ri1])
    plt.hist(data_df[data_df['K_r phys'] == spring_const_vals['K_r phys'].iloc[ri1]]['bond lengths'].iloc[0][0,:], bins=200, density=True, alpha=0.5, label=f"k_r = {spring_const_vals['K_r phys'].iloc[ri1]} lj")
    plt.plot(x, r/Z, label=f"k_r = {spring_const_vals['K_r phys'].iloc[ri1]} lj theory")

    r = np.array([Zr_bond(i, spring_const_vals['K_r phys'].iloc[ri2]) for i in x])
    Z, err = integrate.quad(Zr_bond, 0, 2, args=spring_const_vals['K_r phys'].iloc[ri2])
    plt.hist(data_df[data_df['K_r phys'] == spring_const_vals['K_r phys'].iloc[ri2]]['bond lengths'].iloc[0][0,:], bins=200, density=True, alpha=0.5, label=f"k_r = {spring_const_vals['K_r phys'].iloc[ri2]} lj")
    plt.plot(x, r/Z, label=f"k_r = {spring_const_vals['K_r phys'].iloc[ri2]} lj theory")
    plt.xlabel('Bond length (lj)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('UU_bond_length_histogram2.eps', bbox_inches='tight')

    plt.show()

main()