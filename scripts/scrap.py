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
r_0 = 2

print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F_LJ} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')

def readfile():
    # Change directory to the project folder
    print(f'current dir: {os.getcwd()}')
    currentdir = os.getcwd()
    path = os.path.join(currentdir, r'output')
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

    print(df)

    df1 = df.groupby(['force']).agg({
        'monomer lengths':       lambda s: np.concatenate(s.to_numpy()),
        # 'angle_deg':     lambda s: np.concatenate(s.to_numpy()),
        'types':         lambda s: np.concatenate([np.asarray(x, float) for x in s], axis=1),
        'bond lengths':  lambda s: np.concatenate([np.asarray(x, float) for x in s], axis=1),
        'monomer lengths': lambda s: np.concatenate([np.asarray(x, float) for x in s])
    }).reset_index()
    return df1



def round_sig(x, sig=2):
    # https://stackoverflow.com/a/3413529
    return round(x, sig-int(floor(log10(abs(x))))-1)

def Zr_bond(r, k, F=0, Unfold = False):
    alpha = (r*F)/(T_LJ)
    
    if Unfold:
        if F != 0:
            if alpha < 1E-6:
                exponent = (2*k/T_LJ)*np.log(1-r**2/4)+np.log(2*np.pi)+alpha**2/6+alpha**4/120
                integrand = r**2*np.exp(exponent)
            elif alpha < 50:
                exponent = (2*k/T_LJ)*np.log(1-r**2/4)+np.log(2*np.pi)+np.log(np.sinh(alpha))-np.log(alpha)
                integrand = r**2*np.exp(exponent)
            else:
                exponent = (2*k/T_LJ)*np.log(1-r**2/4)+np.log(2*np.pi)+alpha-np.log(2*alpha)
                integrand = r**2*np.exp(exponent)
            
        else:
            integrand = r**2*(((1-(r**2)/4)**(2*k/T_LJ)))
    else: 
        
        if F != 0:
            if alpha < 1E-6:
                exponent = (-k_r/(2*T_LJ))*(r-1)**2 + np.log(2*np.pi)+alpha**2/6+alpha**4/120
                integrand = r**2*np.exp(exponent)
            elif alpha < 50:
                exponent = (-k_r/(2*T_LJ))*(r-1)**2 + np.log(2*np.pi)+np.log(np.sinh(alpha))-np.log(alpha)
                integrand = r**2*np.exp(exponent)
            else:
                exponent = (-k_r/(2*T_LJ))*(r-1)**2 + np.log(2*np.pi)+alpha-np.log(2*alpha)
                integrand = r**2*np.exp(exponent)
        else:
            exponent = (-k_r/(2*T_LJ))*(r-1)**2
            integrand = r**2*np.exp(exponent)
    #integrand = r**2*np.exp(-U_eff(r,k,F))
    return integrand

def U_eff(r,k,F):
    alpha = (r*F)/(2*T_LJ)
    return -(2*k/T_LJ)*np.log(1-(r**2)/4)-np.log((2*np.pi*np.sinh(alpha)/alpha))

def ExpectedRIntegral_bond(r, k, F, Unfold = False):
    return Zr_bond(r,k,F,Unfold)*r

def ExpectedRSqrIntegral_bond(r, k, F, Unfold = False):
    return Zr_bond(r,k,F,Unfold)*r**2

def ExpectedR(k,F,Unfold=False):
    if Unfold:
        Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k,F,Unfold))
        integral , err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k,F,Unfold))
    else:
        Z, Zerr = integrate.quad(Zr_bond, 0, np.inf, args=(k,F,Unfold))
        integral , err = integrate.quad(ExpectedRIntegral_bond, 0, np.inf, args=(k,F,Unfold))
    
    invZ = 1/Z
    expectedR_new = invZ*integral
    # invZerr = Zerr/(Z**2)

    
    return expectedR_new

def ExpectedRSqr(k,F,Unfold=False):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k,F,Unfold))
    invZ = 1/Z
    # invZerr = Zerr/(Z**2)

    integral1, err = integrate.quad(ExpectedRSqrIntegral_bond, 0, 2, args=(k,F,Unfold))
    integral2, err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k,F,Unfold))
    expectedR_sqr_new = invZ*integral1*2 + 2*(invZ*integral2)**2

    return expectedR_sqr_new

def VarR(k,F,Unfold=False):
    
    expectedR = ExpectedR(k,F,Unfold)
    expectedRSqr = ExpectedRSqr(k,F,Unfold)

    var = expectedRSqr-expectedR**2
    return var

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
            sem_mlen.append((np.mean(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['force'] == force]['monomer lengths'].iloc[0])))
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
    # F = np.linspace(0,700,1000)
    # x_FF = np.array([ExpectedR(k_r,i,Unfold=False) for i in F])
    # # x_FF = (F/k_r + 1)
    # x_UU = np.array([ExpectedR(k_r_prime,i,Unfold=True) for i in F])
    # # x_UU = 2*np.sqrt(1-np.exp(-F/(2*k_r_prime)))
    # x_FU = x_FF + x_UU

    # x_FF *= 2
    # x_UU *= 2

    # x_0 = ExpectedR(k_r,0,Unfold=False)

    os.chdir('..')
    # os.chdir('..')
    currentdir = os.getcwd()
    path = os.path.join(currentdir, r'figures')

    blen_arr = np.array(data_df['bond lengths'].iloc[0])
    btype_arr = np.array(data_df['types'].iloc[0])

    print('--------------------------------------')
    print(blen_arr[0,:].shape[0])
    print(blen_arr[0,:])

    x = np.linspace(0,blen_arr[0,:].shape[0], blen_arr[0,:].shape[0])
    plt.figure(1)
    sc = plt.scatter(
        x,
        blen_arr[0,:],
        s=2.5,
        alpha=0.7,
        c=btype_arr[0,:],
        cmap='viridis',
        label=f'bond 1'
    )
    #plt.plot(blen_arr[0,:])
    plt.title('bond 1')
    plt.xlabel('index')
    plt.ylabel('length')
    cbar = plt.colorbar(sc)
    cbar.set_label('Type')

    plt.figure(2)
    sc = plt.scatter(
        x,
        blen_arr[1,:],
        s=2.5,
        alpha=0.7,
        c=btype_arr[1,:],
        cmap='viridis',
        label=f'bond 2'
    )
    #plt.plot(blen_arr[1,:])
    plt.title('bond 2')
    plt.xlabel('index')
    plt.ylabel('length')
    cbar = plt.colorbar(sc)
    cbar.set_label('Type')

    plt.figure(3)
    mlen_arr = data_df['monomer lengths'].iloc[0]
    mtype_arr = btype_arr[0,:] + btype_arr[1,:]

    print('______________________')
    print(mlen_arr.shape)
    print(blen_arr.shape)
    sc = plt.scatter(
        x,
        mlen_arr,
        s=2.5,
        alpha=0.7,
        c=mtype_arr,
        cmap='viridis',
        label=f'bond 1'
    )
    plt.title('monomer length')
    plt.xlabel('index')
    plt.ylabel('length')
    cbar = plt.colorbar(sc)
    cbar.set_label('Type')

    plt.show()
main()