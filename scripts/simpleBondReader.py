print('start')
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re as re
from scipy import constants
print('imported')

# Point of script:
# One/two bond unfolded fractions
# Mean time one/two unfolded + std/se
# Free energy + standard error

# Constants
timestep = 0.002 # lj
nevery = 5 # how often data is dumped
indextime = timestep * nevery # time between each data point in picoseconds

sigma = 22.5 # nm
epsilon = 225E-19 # J
mass = 2.12E-22 # kg
F = (epsilon/sigma)*(10**18) # pN
Temp = epsilon/constants.Boltzmann
tau = (sigma**-9)*np.sqrt(mass/epsilon)

T_LJ = 0.1840865
chosenForce = 0.0

def readfile():
    # Change directory to the project folder
    os.chdir(r'/home/jacob/projects/Fibrin-Monomer/output')
    print(os.getcwd())
    print(os.listdir())

    folder_lst = sorted(os.listdir())
    rows = []

    folder_num = len(folder_lst)
    counter = 1
    for folder in folder_lst:
        # https://docs.python.org/3/library/re.html
        # https://docs.python.org/3/howto/regex.html#regex-howto
        force_match = re.search(f'Force{chosenForce}?', folder)
        seed_match = re.search(r'Seed(\d+)', folder)

        if not (force_match and seed_match):
            print('none found!')
            continue

        force = float(force_match.group(1))
        seed = int(seed_match.group(1))

        folder_path = os.path.join(os.getcwd(), folder)
        os.chdir(folder_path)

        btype_lst = [[],[]]
        blen_lst = [[],[]]

        with open('bondlengths.dump') as f:
            isItem = False
            counter2 = 0
            for line in f:
                if isItem:
                    btype_lst[counter2].append(line[0])
                    blen_lst[counter2].append(line[2::].strip(' \n'))
                    counter2 += 1
                if counter2 == 2:
                    isItem = False
                    counter2 = 0
                if 'ENTRIES c_btype c_bondlen' in line:
                    isItem = True
        
        row = {
            "force": force,
            "seed": seed,
            "types": np.array(btype_lst, dtype=int),
            "lengths": np.array(blen_lst, dtype=np.float64)
        }
        rows.append(row)
        os.chdir('..')
        print(f'{counter}/{folder_num} folders read')
        counter += 1

    df = pd.DataFrame(rows)

    return df

def compute(btype_arr, blen_arr):
    # Calculate:
    # Mean extension
    # One/two bond unfolded fractions
    # Mean time one/two unfolded + std/se
    # Free energy + standard error

    mean_extension = np.mean(blen_arr[0]+blen_arr[1]-blen_arr[0][0]-blen_arr[1][0])

    timestep_no = len(btype_arr[1])

    #######################################################################################
    # unfolded fractions

    UU_count = 0
    FU_count = 0
    for i in range(timestep_no):
        if btype_arr[0][i] == 2 and btype_arr[1][i] == 2:
            UU_count += 1
        elif btype_arr[0][i] == 2 or btype_arr[1][i] == 2:
            FU_count += 1

    UU_fraction = UU_count / btype_arr.shape[1]
    FU_fraction = FU_count / btype_arr.shape[1]
    FF_fraction = 1 - (UU_fraction + FU_fraction)

    #######################################################################################
    # unfold times
    FU_time_lst = []
    UU_time_lst = []
    counter = 0
    FU = False
    UU = False
    for i in range(timestep_no):
        if btype_arr[0][i] == 2:
            if btype_arr[1][i] == 2 and FU:
                    FU = False
                    UU = True
                    FU_time_lst.append(counter)
                    counter = 0
            elif btype_arr[1][i] == 1 and not FU:
                FU = True
                if UU:
                    UU = False
                    UU_time_lst.append(counter)
                    counter = 0
        if btype_arr[1][i] == 2:
            if btype_arr[0][i] == 1 and not FU:
                FU = True
                if UU:
                    UU = False
                    UU_time_lst.append(counter)
                counter = 0
        if FU or UU:
            counter += 1
        if i == timestep_no-1:
            if FU:
                FU_time_lst.append(counter)
            elif UU:
                UU_time_lst.append(counter)

    FU_time_arr = np.array(FU_time_lst, dtype= float)
    UU_time_arr = np.array(UU_time_lst, dtype= float)

    FU_mean_duration = np.mean(FU_time_arr*indextime)
    FU_mean_duration_std = np.std(FU_time_arr*indextime)
    FU_mean_duration_se = FU_mean_duration_std / np.sqrt(len(FU_time_lst))

    UU_mean_duration = np.mean(UU_time_arr*indextime)
    UU_mean_duration_std = np.std(UU_time_arr*indextime)
    UU_mean_duration_se = UU_mean_duration_std / np.sqrt(len(UU_time_lst))

    ############################################################################################
    # Equilibrium constant and free energy calculations
    # K_eq = (fraction of folded bs) / (fraction of unfolded bs)
    # Free energy G/kT = ln(K_eq). 
    # This is the difference in free energy between folded and unfolded states.

    if FF_fraction > 0:
        K_eq_UU = UU_fraction / FF_fraction
        K_eq_FU = FU_fraction / FF_fraction
    else:
        K_eq_FU = 10e40
        K_eq_UU = 10e40
    return mean_extension, FU_fraction, UU_fraction, FU_mean_duration, FU_mean_duration_std, FU_mean_duration_se,\
        UU_mean_duration, UU_mean_duration_std, UU_mean_duration_se, K_eq_FU, K_eq_UU

def weighted_avg_and_se(group, mean_col, se_col):
    means = group[mean_col].values
    ses = group[se_col].values
    weights = 1 / ses**2
    weighted_mean = np.sum(weights * means) / np.sum(weights)
    weighted_se = np.sqrt(1 / np.sum(weights))
    return pd.Series({'mean': weighted_mean, 'sem': weighted_se})

def computeSeedAveraged(data_df):
    # Equilibrium constant and free energy calculations
    # K_eq = (fraction of folded bs) / (fraction of unfolded bs)
    # Free energy G/kT = ln(K_eq). 
    # This is the difference in free energy between folded and unfolded states.

    
    data_force_groups = data_df.groupby(['force'], group_keys=True)
    # print(data_df.columns)
    # print(data_df.head)
    # print('------------------------------------')
    # print(data_force_groups)
    
    analysis_df = data_force_groups.agg({
        'mean extension': ['mean', 'sem'],
        'FU fraction': ['mean', 'sem'],
        'UU fraction': ['mean', 'sem'],
        'FU equilibrium constant': ['mean', 'sem'],
        'UU equilibrium constant': ['mean', 'sem']
    })
    
    # print(analysis_df)
    
    FU_duration = data_df.groupby('force').apply(
    lambda g: weighted_avg_and_se(g, 'FU mean duration', 'FU mean duration se')
    )
    UU_duration = data_df.groupby('force').apply(
        lambda g: weighted_avg_and_se(g, 'UU mean duration', 'UU mean duration se')
    )

    analysis_df[('FU mean duration', 'mean')] = FU_duration['mean']
    analysis_df[('FU mean duration', 'sem')]   = FU_duration['sem']
    analysis_df[('UU mean duration', 'mean')] = UU_duration['mean']
    analysis_df[('UU mean duration', 'sem')]   = UU_duration['sem']

    # pull out the two columns
    eqc = analysis_df[('FU equilibrium constant', 'mean')]
    sem = analysis_df[('FU equilibrium constant', 'sem')]

    # value: −ln(K), but only where K>0
    fe_val = -np.log(eqc.where(eqc > 0, np.nan))

    # se: sem(K)/K, but only where K>0
    fe_se = (sem / eqc).where(eqc > 0, np.nan)

    # assign back
    analysis_df[('FU free energy', 'value')] = fe_val
    analysis_df[('FU free energy', 'sem')]    = fe_se
    
     # pull out the two columns
    eqc = analysis_df[('UU equilibrium constant', 'mean')]
    sem = analysis_df[('UU equilibrium constant', 'sem')]

    # value: −ln(K), but only where K>0
    fe_val = -np.log(eqc.where(eqc > 0, np.nan))

    # se: sem(K)/K, but only where K>0
    fe_se = (sem / eqc).where(eqc > 0, np.nan)

    # assign back
    analysis_df[('UU free energy', 'value')] = fe_val
    analysis_df[('UU free energy', 'sem')]    = fe_se

    # analysis_df['FU free energy'] = analysis_df.agg(
    #     value, std
    # )
    
    # analysis_df[('FU equilibrium constant', 'mean')].apply(
    #     lambda x: -np.log(x) if x > 0 else np.inf
    # )
    # analysis_df['FU free energy', 'se'] = analysis_df['FU equilibrium constant'].agg(sem)
    # .apply(
    #     lambda x: x['FU equilibrium constant', 'sem']/x['FU equilibrium constant', 'mean'] \
    #     if x['FU equilibrium constant', 'mean'] > 0 else np.nan
    # )
    
    # analysis_df['UU free energy', 'value'] = analysis_df['UU equilibrium constant', 'mean'].apply(
    #     lambda x: -np.log(x) if x > 0 else np.inf
    # )
    # analysis_df['UU free energy', 'se'] = analysis_df.apply(
    #     lambda x: x['UU equilibrium constant', 'sem']/x['UU equilibrium constant', 'mean'] \
    #     if x['UU equilibrium constant', 'mean'] > 0 else np.nan
    # )

    return analysis_df

def plotgraphs(data_df, monomer_df):
    # Plotting the b lengths over time
    os.chdir(r'\\wsl.localhost\Ubuntu\home\jacob\projects\Fibrin-Monomer') 

    t = np.linspace(0, blen_arr.shape[1]-1, blen_arr.shape[1])*tau

    plt.figure(1)
    plt.scatter(t, blen_arr[0], label='Bond type. Purple: folded, Yellow: unfolded', c = btype_arr[0])
    plt.plot(t, blen_arr[0])

    plt.xlabel('Time (ns)')
    plt.ylabel('Bond Length (nm)')   
    plt.title(f'Bond 1 Lengths/Time With Force {force} (pN)')
    plt.legend()

    plt.savefig(f'figures/b1Lengths_Force{force}_Seed{seed}.png', dpi=600, bbox_inches='tight')


    plt.figure(2)

    plt.scatter(t, blen_arr[1], label='Bond type. Purple: folded, Yellow: unfolded', c = btype_arr[1])
    plt.plot(t, blen_arr[1])

    plt.xlabel('Time (ns)')
    plt.ylabel('Bond Length (nm)')
    plt.title(f'Bond 2 Lengths/Time With Force {force} (pN)')
    plt.legend()

    plt.savefig(f'figures/b2lengths_Force{force}_Seed{seed}.png', dpi=600, bbox_inches='tight')

    plt.figure(3)
    plt.plot(t, blen_arr[0]+blen_arr[1]-blen_arr[0][0]-blen_arr[1][0], label=f'Mean Extension = {mean_extension}')

    plt.xlabel('Time (ns)')
    plt.ylabel('Molecule Extension (nm)')
    plt.title(f'Molecule Extension/Time With Force {force} (pN)')
    plt.legend()

    plt.savefig(f'figures/molecule_extension_Force{force}_Seed{seed}.png', dpi=600, bbox_inches='tight')

    plt.show()

def main():
    data_df = readfile()
    dupes = data_df.duplicated(subset=['force', 'seed'])
    print(data_df[dupes])
    rows = []
    force_seed_pairs = data_df[['force', 'seed']].drop_duplicates().values
    for force, seed in force_seed_pairs:
        
        mean_extension, FU_fraction, UU_fraction, FU_mean_duration, FU_mean_duration_std, \
        FU_mean_duration_se, UU_mean_duration, UU_mean_duration_std, UU_mean_duration_se, \
        K_eq_FU, K_eq_UU \
        = compute(data_df.loc[(data_df['force'] == force) & (data_df['seed'] == seed), 'types'].iloc[0], \
                data_df.loc[(data_df['force'] == force) & (data_df['seed'] == seed), 'lengths'].iloc[0])
        
        row = {
        "force": force * F,
        "seed": seed,
        "mean extension": mean_extension * sigma,
        "FU fraction": FU_fraction,
        "UU fraction": UU_fraction,
        "FU mean duration": FU_mean_duration * tau,
        "FU mean duration std": FU_mean_duration_std * tau,
        "FU mean duration se": FU_mean_duration_se * tau,
        "UU mean duration": UU_mean_duration * tau,
        "UU mean duration std": UU_mean_duration_std * tau,
        "UU mean duration se": UU_mean_duration_se * tau,
        "FU equilibrium constant": K_eq_FU,
        "UU equilibrium constant": K_eq_UU
        }
        rows.append(row)

    monomer_df = pd.DataFrame(rows)
    monomer_df = computeSeedAveraged(monomer_df)
    os.chdir('..')
    path = os.getcwd()
    os.chdir(path, r'/output')
    monomer_df.to_csv(f'Force{monomer_df['force'].iloc(0)}Data.csv')

    plotgraphs(data_df, monomer_df)

main()