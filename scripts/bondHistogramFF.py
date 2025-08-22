# Function of script:
# 1) If VaryForce global parameter is True, create histograms that show folded distribution shifts with force applied
# 2) If VaryForce global parameter is False, create histograms that show folded distribution shifts with varying spring constant values.
#       Constant force = 0.
#       Angular spring constant is found here aswell.

# Data in output must reflect these changes, and data folder titling must be as follows:
# 1) VaryForce: 
#       Use run_fibrin.sh to generate a folder titled ForceSeedFF. 
#       Its contents must have data stored in folders of the format: Force{force}_Seed{seed}, force must be a decimal.
# 2) Not VaryForce: 
#       Use run_fibrinff.sh to generate a folder titled K_rt_SeedFF. 
#       Its contents have data stored in folders of the format: Seed{seed}_k_r{k_r}_k_theta{k_theta}, all must be decimals.

# Be sure to disable reactions in fibrin-monomer.lam and set both bond types to 1 in initial_configuration.txt

# Need to fix radial theory std/mean.

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

k_theta = 7.785
k_r = 43.87
k_r_prime = 3
r_0 = 1
r_0_prime = 2

VaryForce = False

print(f'current dir: {os.getcwd()}')
print(f'sigma: {sigma} nm, epsilon: {epsilon} J, mass: {mass} kg, tau: {tau} s,') 
print(f'F: {F} pN, F without conversion: {epsilon/(sigma*10**-9)} pN, tau: {tau}')
print(f'T_LJ: {T_LJ} lj, Temp: {Temp} K, boltzmann constant: {constants.k} J/K')

def readfile():
    df1 = pd.DataFrame()
    if VaryForce:
        print(f'current dir: {os.getcwd()}')
        currentdir = os.getcwd()
        print(os.listdir())
        path = os.path.join(currentdir, r'output/ForceSeedFF')
        # path = r"\\wsl$\Ubuntu\home\jacob\projects\Fibrin-Monomer\output\ForceSeedFF"
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

        df1 = df.groupby(['force']).agg({
            'monomer lengths':       lambda s: np.concatenate(s.to_numpy()),
            'angle_deg':     lambda s: np.concatenate(s.to_numpy()),
            'types':         lambda s: np.concatenate(s.to_numpy()),
            'bond lengths':  lambda s: np.concatenate([np.asarray(x, float) for x in s], axis=1),
        }).reset_index()
    else:
        # Change directory to the project folder
        print(os.getcwd())
        os.chdir('..')
        print(os.getcwd())
        path = os.path.join(os.getcwd(), r'Fibrin-Monomer/output/K_rt_SeedFF')
        # path = r'\\wsl.localhost\Ubuntu\home\jacob\projects\LammpsCode\output\CORRff'
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
            r_match = re.search(r'k_r(\d+(?:\.\d+)?)', folder)
            t_match = re.search(r'k_theta(\d+(?:\.\d+)?)', folder)

            if not (r_match and t_match and seed_match):
                continue
            
            seed = int(seed_match.group(1))
            k_r_lmmps = float(r_match.group(1))
            k_t_lmmps = float(t_match.group(1))

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

                "K_r phys": k_r_lmmps*2,
                "K_theta phys": k_t_lmmps*2,

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
            df1 = df.groupby(['K_r phys', 'K_theta phys']).agg({
                'monomer lengths':       lambda s: np.concatenate(s.to_numpy()),
                'angle_deg':     lambda s: np.concatenate(s.to_numpy()),
                'types':         lambda s: np.concatenate(s.to_numpy()),
                'bond lengths':  lambda s: np.concatenate([np.asarray(x, float) for x in s], axis=1),
            }).reset_index()
    return df1

def round_sig(x, sig=2):
    # https://stackoverflow.com/a/3413529
    return round(x, sig-int(floor(log10(abs(x))))-1)

def ZTheta(theta, k_t):
    exponent = (-k_t/(2*T_LJ))*(theta-np.pi)**2
    integrand =  np.sin(theta)*np.exp(exponent)
    return integrand

def Zr_bond(r, k, F=0):
    alpha = (r*F)/(T_LJ)
    if F != 0:
        exponent = (-k_r/(2*T_LJ))*(r-1)**2
        integrand = r**2*np.exp(exponent)*(2*np.pi*sinhc(alpha))
    else:
        exponent = (-k_r/(2*T_LJ))*(r-1)**2
        integrand = r**2*np.exp(exponent)
    #integrand = r**2*np.exp(-U_eff(r,k,F))
    return integrand

def sinhc(x):
    if np.isclose(x, 0.0):
        return 1.0 + x**2/6.0  # series expansion
    else:
        return np.sinh(x)/x

def ExpectedThetaIntegral(theta, k_t):
    return theta*ZTheta(theta,k_t)

def ExpectedTheta(k_t):
    Z, Zerr = integrate.quad(ZTheta, 0, np.pi, args=(k_t,))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedThetaIntegral, 0, np.pi, args=(k_t,))
    expectedTheta = invZ*integral
    expectedTheta_err = expectedTheta*(invZerr/invZ + err/integral)
    return expectedTheta, expectedTheta_err

def ExpectedRIntegral_bond(r, k, F=0):
    return Zr_bond(r,k,F)*r

def ExpectedRSqrIntegral_bond(r, k, F=0):
    return Zr_bond(r,k,F)*r**2

def ExpectedR(k,F=0):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k,F))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral , err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k,F))
    expectedR_new = invZ*integral*2

    error = err*invZerr*2
    return expectedR_new, error

def ExpectedRSqr(k,F=0):
    Z, Zerr = integrate.quad(Zr_bond, 0, 2, args=(k,F))
    invZ = 1/Z
    invZerr = Zerr/(Z**2)

    integral1, err = integrate.quad(ExpectedRSqrIntegral_bond, 0, 2, args=(k,F))
    integral2, err = integrate.quad(ExpectedRIntegral_bond, 0, 2, args=(k,F))
    expectedR_sqr_new = invZ*integral1*2 + 2*(invZ*integral2)**2

    return expectedR_sqr_new

def VarR(k,F=0):
    
    expectedR, error = ExpectedR(k,F)
    expectedRSqr = ExpectedRSqr(k,F)

    var = expectedRSqr-expectedR**2
    return var

def stats(data_df):
    if VaryForce:
        rt_pairs = data_df[['force']].drop_duplicates().values
    else:
        rt_pairs = data_df[['K_r phys', 'K_theta phys']].drop_duplicates().values
    

    std_mlen = []
    sem_std_mlen = []
    mean_mlen = []
    sem_mlen = []
    mean_t = []
    sem_t = []
    std_t = []
    sem_std_t = []
    mean_blen = []

    if VaryForce:
        for forces in rt_pairs:
            force = forces[0]
            try:
                std_mlen.append(np.std(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))
                sem_std_mlen.append((np.std(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['force'] == force]['monomer lengths'].iloc[0])))
                
                mean_mlen.append(np.mean(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))
                sem_mlen.append((np.mean(data_df[data_df['force'] == force]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[data_df['force'] == force]['monomer lengths'].iloc[0])))
                mean_blen.append(np.mean(np.array(data_df[data_df['force'] == force]['bond lengths'].iloc[0]), axis=1))

            except ValueError as e:
                print(f"Error calculating stats for force={force}: {e}")
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

            "lengths": data_df['monomer lengths'].values,
            "angles": data_df['angle_deg'].values,
        })
    else:
        for k_r, k_t in rt_pairs:
            try:
                std_mlen.append(np.std(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['monomer lengths'].iloc[0]))
                sem_std_mlen.append((np.std(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['monomer lengths'].iloc[0])))
                
                mean_mlen.append(np.mean(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['monomer lengths'].iloc[0]))
                sem_mlen.append((np.mean(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['monomer lengths'].iloc[0]))/np.sqrt(len(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['monomer lengths'].iloc[0])))

                std_t.append(np.std(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['angle_deg'].iloc[0])*np.pi/180)
                sem_std_t.append((np.std(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['angle_deg'].iloc[0])*np.pi/180)/np.sqrt(len(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['angle_deg'].iloc[0])))

                mean_t.append(np.mean(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['angle_deg'].iloc[0])*np.pi/180)
                sem_t.append((np.mean(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['angle_deg'].iloc[0])*np.pi/180)/np.sqrt(len(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['angle_deg'].iloc[0])))

                mean_blen.append(np.max(np.array(data_df[(data_df['K_r phys'] == k_r) & (data_df['K_theta phys'] == k_t)]['bond lengths'].iloc[0]), axis=1))

            except ValueError as e:
                print(f"Error calculating std for k_r={k_r}, k_t={k_t}: {e}")
                std_mlen.append(np.nan)
                sem_std_mlen.append(np.nan)

                mean_mlen.append(np.nan)
                mean_mlen.append(np.nan)

                mean_t.append(np.nan)
                sem_t.append(np.nan)

        mean_t_arr = np.array(mean_t, dtype=np.float64)

        delta_t = np.abs(np.pi-mean_t_arr-0.1)

        print(f'index closest to 10%: {delta_t.argmin()}')
        print(f'mean_t: {mean_t_arr[delta_t.argmin()]}')
        print(f'k_r: {rt_pairs[delta_t.argmin()][0]}, k_t: {rt_pairs[delta_t.argmin()][1]}')

        monomer_df = pd.DataFrame({
            "K_r": data_df['K_r phys'],
            "K_theta": data_df['K_theta phys'],

            "monomer length mean": mean_mlen,
            "monomer length sem": sem_mlen,
            "monomer length std": std_mlen,
            "monomer length std sem": sem_std_mlen,

            "theta mean": mean_t_arr,
            "theta sem": sem_t,
            "theta std": std_t,
            "theta std sem": sem_std_t,

            "blen mean": mean_blen,

            "lengths": data_df['monomer lengths'].values,
            "angles": data_df['angle_deg'].values,
        })
   
    return monomer_df 

def main():
    data_df = readfile()
    if VaryForce:
        data_df.sort_values(by=['force'], inplace=True)
    else:
        data_df.sort_values(by=['K_r phys', 'K_theta phys'], inplace=True)
    
    monomer_df = stats(data_df)

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

    intlim = [0,5*r_0]
    fig, ax = plt.subplots()
    ax.set_box_aspect(2/3)
    

    if VaryForce:
        x = np.linspace(0,3,200)
        plt.figure(1)
        for j in range(0,5):
            i = 2*j
            F = monomer_df['force'].iloc[i]
            r = np.array([Zr_bond(i, k_r, F) for i in x])
            Z, err = integrate.quad(Zr_bond, intlim[0], intlim[1], args=(k_r_prime, F))
            plt.plot(x, r/Z, label=f"force = {F} lj theory")  
            plt.hist(data_df['bond lengths'].iloc[i][1,:], bins=500, density=True, alpha=0.5, label=f"force = {F} lj")
            plt.xlabel('Bond 2 length (lj)')
        plt.ylabel('Frequency')
        plt.title(f'Single bond length histogram, force dependent, k_r = {k_r}, k_theta = {k_theta}')
        plt.legend()
        plt.savefig(f'U_bond_length_histogram_k{k_r}.eps', bbox_inches='tight')
        plt.savefig(f'U_bond_length_histogram_k{k_r}.png', bbox_inches='tight')
    else:
        spring_const_vals = data_df[['K_r phys', 'K_theta phys']].drop_duplicates()
      
        print('0 FORCE PLOTS')


        ri1 = round(len(spring_const_vals['K_r phys'].drop_duplicates())/2)
        ti1 = round(len(spring_const_vals['K_theta phys'].drop_duplicates())/2)

        k_arr = np.linspace(1, 100, 1000)

        R_var = np.array([VarR(i) for i in k_arr])
        R_exp = np.array([ExpectedR(i) for i in k_arr])
        Theta_exp = np.array([ExpectedTheta(i) for i in k_arr])[:,0]

        R_std_ratio = np.sqrt(R_var)/R_exp
        Theta_exp_ratio = (np.pi-Theta_exp)/np.pi

        r_std_arr = monomer_df[(monomer_df['K_theta'] == spring_const_vals['K_theta phys'].iloc[ti1])]\
                 ['monomer length std'].values
        r_mean_arr = monomer_df[(monomer_df['K_theta'] == spring_const_vals['K_theta phys'].iloc[ti1])]\
                 ['monomer length mean'].values
        
        t_mean_arr = monomer_df[(monomer_df['K_r'] == spring_const_vals['K_r phys'].iloc[ri1])]\
                 ['theta mean'].values
        t_mean_arr_ratio = (np.pi-t_mean_arr)/np.pi

        print(t_mean_arr)

        fig, ax = plt.subplots()
        ax.set_box_aspect(2/3)


        plt.figure(1)

        plt.errorbar(
            monomer_df[(monomer_df['K_r'] == spring_const_vals['K_r phys'].iloc[ri1])]\
                 ['K_theta'],
            t_mean_arr_ratio, 
            yerr=monomer_df[(monomer_df['K_r'] == spring_const_vals['K_r phys'].iloc[ri1])]\
                 ['theta sem'].values/180,
            ecolor='black',
            fmt='o',   
            markersize=3,
            capsize=4,
            capthick=0.5,
            elinewidth=0.3
        )
        plt.plot(k_arr, 
                 Theta_exp_ratio, 
                 label=f'k_theta = {spring_const_vals["K_theta phys"].iloc[ri1]} lj theory'
        )

        plt.title(f"Monomer angle / pi histogram at k_r = {spring_const_vals['K_r phys'].iloc[ri1]} lj. \nTimestep=0.001, Count=800000, 5 seeds")
        plt.xlabel('Monomer angle (rad)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('mean_theta_vs_K_theta.eps', bbox_inches='tight')
        plt.savefig('mean_theta_vs_K_theta.png', bbox_inches='tight')

        plt.figure(2)
        plt.plot(
            k_arr, 
            R_std_ratio, 
            label='theory std/<r>'
        )

        plt.errorbar(
            monomer_df[(monomer_df['K_theta'] == spring_const_vals['K_theta phys'].iloc[ti1])]\
                 ['K_r'],
            r_std_arr/r_mean_arr,
            yerr=monomer_df[(monomer_df['K_theta'] == spring_const_vals['K_theta phys'].iloc[ti1])]['monomer length sem']/ \
                monomer_df[(monomer_df['K_theta'] == spring_const_vals['K_theta phys'].iloc[ti1])]['monomer length mean'],
            ecolor='black',
            fmt='o',   
            markersize=3,
            capsize=4,
            capthick=0.5,
            elinewidth=0.3
        )
        plt.xlabel('K_r')
        plt.ylabel('Monomer length std / <R> (lj)')
        plt.title('Monomer length fluctuation')
        plt.legend()
        plt.savefig('std_r_vs_K_r.eps', bbox_inches='tight')
        plt.savefig('std_r_vs_K_r.png', bbox_inches='tight')

    plt.show()

main()
