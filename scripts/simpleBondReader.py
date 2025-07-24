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
timestep = 0.0001 # lj
nevery = 5 # how often data is dumped
indextime = timestep * nevery # time between each data point in picoseconds

sigma = 22.5E-9 # m
epsilon = 2.25E-20 # J
mass = 2.12E-22 # kg
F = (epsilon/sigma) # N
T_LJ = 1.0 #18.4
Temp = T_LJ*epsilon/constants.Boltzmann # K
tau = (sigma)*np.sqrt(mass/epsilon) # s

chosenForce = '0.0'

def readfile():
    # Change directory to the project folder
    path = os.path.join(os.getcwd(), r'output')
    os.chdir(path)
    print(os.getcwd())
    

    folder_lst = sorted(os.listdir())
    rows = []

    folder_num = len(folder_lst)
    counter = 1
    for folder in folder_lst:
        # https://docs.python.org/3/library/re.html
        # https://docs.python.org/3/howto/regex.html#regex-howto
        force_match = re.search(f'Force{chosenForce}', folder)
        seed_match = re.search(r'Seed(\d+)', folder)
        

        if not (force_match and seed_match):
            continue

        print(force_match)
        print(seed_match)

        force = float(chosenForce)
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
        if btype_arr[0][i] != btype_arr[1][i]:
            if not FU and not UU:
                FU = True
                counter = 0
            elif UU:
                FU = True
                UU = False
                UU_time_lst.append(counter)
                counter = 0

        elif btype_arr[0][i] == 2 and btype_arr[1][i] == 2:
            if not FU and not UU:
                UU = True
                counter = 0
            elif FU:
                UU = True
                FU = False
                FU_time_lst.append(counter)
                counter = 0

        elif btype_arr[0][i] == 1 and btype_arr[1][i] == 1:
            if FU:
                FU = False
                FU_time_lst.append(counter)
                counter = 0
            elif UU:
                UU = False
                UU_time_lst.append(counter)
                counter = 0
        
        if FU or UU:
            counter += 1

    if FU:
        FU_time_lst.append(counter)
    elif UU:
        UU_time_lst.append(counter)

    FU_time_arr = np.array(FU_time_lst, dtype= float)
    UU_time_arr = np.array(UU_time_lst, dtype= float)

    # if no values in the list, sem will be zero. Set sem to timestep to avoid division by 
    # zero, this should be physically meaningful.
    if len(FU_time_arr) > 0:
        FU_mean_duration = np.mean(FU_time_arr * indextime)
        FU_mean_duration_std = np.std(FU_time_arr * indextime)
        FU_mean_duration_se = FU_mean_duration_std / np.sqrt(len(FU_time_lst))
    else:
        FU_mean_duration = 0
        FU_mean_duration_std = 0
        FU_mean_duration_se = timestep

    if len(UU_time_arr) > 0:
        UU_mean_duration = np.mean(UU_time_arr * indextime)
        UU_mean_duration_std = np.std(UU_time_arr * indextime)
        UU_mean_duration_se = UU_mean_duration_std / np.sqrt(len(UU_time_lst))
    else:
        UU_mean_duration = 0
        UU_mean_duration_std = 0
        UU_mean_duration_se = timestep

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

def weighted_avg_std_se(group, mean_col, se_col, std_col):
    # weighted average and standard error of the mean
    # pooled standard deviation. https://www.statisticshowto.com/pooled-standard-deviation

    mean = group[mean_col].values
    std = group[std_col].values
    se = group[se_col].values

    valid = (~np.isnan(mean)) & (~np.isnan(std)) & (se > 0) & (~np.isnan(se))
    if np.any(~valid):
        print(f"Valid mean/std/sem NaN or sem > 0 entries: {np.sum(valid)} out of {len(valid)}")

    if np.sum(valid) == 0:
        return pd.Series({'mean': np.nan, 'std': np.nan, 'sem': np.nan})
    
    mean = mean[valid]
    std = std[valid]
    se = se[valid]

    weights = 1 / se**2
    weighted_mean = np.sum(weights * mean) / np.sum(weights)
    weighted_se = np.sqrt(1 / np.sum(weights))
    weighted_std = np.sqrt(np.sum((std**2)) / (len(std)))  # pooled standard deviation

    return pd.Series({'mean': weighted_mean, 'std': weighted_std, 'sem': weighted_se})

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
        'mean extension': ['mean', 'std', 'sem'],
        'FU fraction': ['mean', 'std', 'sem'],
        'UU fraction': ['mean', 'std', 'sem'],
        'FU equilibrium constant': ['mean', 'std', 'sem'],
        'UU equilibrium constant': ['mean', 'std', 'sem']
    })

    FU_duration = data_df.groupby('force').apply(
        lambda group: weighted_avg_std_se(
            group, 'FU mean duration', 'FU mean duration se', 'FU mean duration std'
        )
    )
    UU_duration = data_df.groupby('force').apply(
            lambda group: weighted_avg_std_se(
                group, 'UU mean duration', 'UU mean duration se', 'UU mean duration std'
        )
    )

    analysis_df[('FU mean duration', 'mean')] = FU_duration['mean']
    analysis_df[('FU mean duration', 'std')]  = FU_duration['std']
    analysis_df[('FU mean duration', 'sem')]   = FU_duration['sem']
    analysis_df[('UU mean duration', 'mean')] = UU_duration['mean']
    analysis_df[('UU mean duration', 'std')]  = UU_duration['std']
    analysis_df[('UU mean duration', 'sem')]   = UU_duration['sem']

    eq_const = analysis_df[('FU equilibrium constant', 'mean')]
    sem = analysis_df[('FU equilibrium constant', 'sem')]

    fe_val = np.where(eq_const > 0, -np.log(eq_const), np.inf)
    fe_se = np.where(eq_const > 0, sem / eq_const, np.nan)

    analysis_df[('FU free energy', 'value')] = fe_val
    analysis_df[('FU free energy', 'sem')]    = fe_se

    eq_const = analysis_df[('UU equilibrium constant', 'mean')]
    sem = analysis_df[('UU equilibrium constant', 'sem')]

    fe_val = np.where(eq_const > 0, -np.log(eq_const), np.inf)
    fe_se = np.where(eq_const > 0, sem / eq_const, np.nan)

    analysis_df[('UU free energy', 'value')] = fe_val
    analysis_df[('UU free energy', 'sem')]    = fe_se

    return analysis_df

def plotgraphs(data_df, monomer_df):
    # Plotting the b lengths over time
    os.chdir(r'..') 
    path = os.path.join(os.getcwd(), r'figures')
    os.chdir(path)
    print(os.getcwd())

    blen_arr_series = data_df['lengths']
    btype_arr_series = data_df['types']

    plt.figure(1)

    for i in range(blen_arr_series.shape[0]):
        blen_arr = blen_arr_series[i][0] * sigma
        btype_arr = btype_arr_series[i][0]
        t = np.arange(len(blen_arr)) * timestep * tau
        plt.plot(t, blen_arr, linewidth=0.7)
        plt.scatter(t, blen_arr, c = btype_arr, s=4, alpha=0.5)

    plt.xlabel('Time (ns)')
    plt.ylabel('Bond Length (nm)')   
    plt.title(f'Bond 1 Lengths/Time With Force {chosenForce.replace('.', '_')} (pN)')

    plt.savefig(f'b1Lengths_Force{chosenForce.replace('.', '_')}.png', dpi=600, bbox_inches='tight')

    plt.figure(2)
    for i in range(blen_arr_series.shape[0]):
        blen_arr = blen_arr_series[i][1] * sigma
        btype_arr = btype_arr_series[i][1]
        t = np.arange(len(blen_arr)) * timestep * tau
        plt.plot(t, blen_arr, linewidth=0.7)
        plt.scatter(t, blen_arr, c = btype_arr, s=4, alpha=0.5)
        

    plt.xlabel('Time (ns)')
    plt.ylabel('Bond Length (nm)')
    plt.title(f'Bond 2 Lengths/Time With Force {chosenForce.replace('.', '_')} (pN)')

    plt.savefig(f'b2lengths_Force{chosenForce.replace('.', '_')}.png', dpi=600, bbox_inches='tight')

    plt.figure(3)
    for i in range(blen_arr_series.shape[0]):
        blen_arr = blen_arr_series[i][0]+blen_arr_series[i][1] * sigma
        t = np.arange(len(blen_arr)) * timestep * tau
        plt.plot(t, blen_arr, linewidth=0.7)

    plt.xlabel('Time (ns)')
    plt.ylabel('Molecule Extension (nm)')
    plt.title(f'Molecule Extension/Time With Force {chosenForce.replace('.', '_')} (pN)')

    plt.savefig(f'molecule_extension_Force{chosenForce.replace('.', '_')}.png', dpi=600, bbox_inches='tight')

    plt.show()


def main():
    data_df = readfile()
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
        if (FU_fraction > 0 and FU_mean_duration == 0) or (UU_fraction > 0 and UU_mean_duration == 0):
            print(f"[WARNING] Fraction > 0 but duration == 0 → Force: {force}, Seed: {seed}")
            print(f"  FU fraction: {FU_fraction}, FU duration: {FU_mean_duration}")
            print(f"  UU fraction: {UU_fraction}, UU duration: {UU_mean_duration}")
        elif (FU_fraction > 0 and np.isnan(FU_mean_duration)) or (UU_fraction > 0 and np.isnan(UU_mean_duration)):
            print(f"[WARNING] Fraction > 0 but duration == NaN → Force: {force}, Seed: {seed}")

        rows.append(row)

    os.chdir('..')
    path = os.path.join(os.getcwd(), r'data files')
    os.chdir(path)
    print(os.getcwd())

    monomer_df = pd.DataFrame(rows)
    monomer_df = computeSeedAveraged(monomer_df)
    monomer_df.to_csv(f'Force{chosenForce}Data.csv')

    plotgraphs(data_df, monomer_df)

main()