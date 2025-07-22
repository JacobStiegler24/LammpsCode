import os
import numpy as np
import matplotlib.pyplot as plt
import re as re

# Constants
timestep = 0.002 # lj
nevery = 5 # how often data is dumped
indextime = timestep * nevery # time between each data point in picoseconds
folderN = 1

sigma = 22.5 # nm
epsilon = 225 # nm*pN
mass = 2.12 # E^-22 kg
tau = sigma*np.sqrt(mass/epsilon)

def readfile():
    # Change directory to the project folder
    os.chdir(r'\\wsl.localhost\Ubuntu\home\jacob\projects\Fibrin-Monomer\output\single')
    print(os.getcwd())

    folder_lst = os.listdir()
    print(folder_lst)
    print(folder_lst[folderN])
    # https://docs.python.org/3/library/re.html
    # https://docs.python.org/3/howto/regex.html#regex-howto
    
    match = re.search(r'Force(\d+\.\d+)?', folder_lst[folderN])
    if match:
        force = float(match.group(1))
    force = force*10*epsilon/sigma

    match = re.search(r'Seed(\d+)?', folder_lst[folderN])
    if match:
        seed = match.group(1)

    print(f'force {force}, seed {seed}')

    os.chdir(rf'\\wsl.localhost\Ubuntu\home\jacob\projects\Fibrin-Monomer\output\single\{folder_lst[folderN]}')

    # two arrays, 2xN in length. column corresponds to bond ID, each row has value at timestep
    btype = [[],[]]
    blen = [[],[]]
    with open('bondlengths.dump') as f:
        isItem = False
        counter = 0
        for line in f:
            if isItem:
                btype[counter].append(line[0])
                blen[counter].append(line[2::].strip(' \n'))
                counter += 1
            if counter == 2:
                isItem = False
                counter = 0
            if 'ENTRIES c_btype c_bondlen' in line:
                isItem = True

    btype_arr = np.array(btype).astype(int)
    blen_arr = np.array(blen).astype(np.float64)

    return btype_arr, blen_arr, force, seed

def printLengthStats(btype_arr, blen_arr):
    b1_meantype = np.mean(btype_arr[0])
    b2_meantype = np.mean(btype_arr[1])

    b1_folded_fraction = 2-np.mean(btype_arr[0])
    b2_folded_fraction = 2-np.mean(btype_arr[1])

    b1_meanlength = np.mean(blen_arr[0])
    b2_meanlength = np.mean(blen_arr[1])

    bothunfolded = 0
    oneunfolded = 0
    for i in range(btype_arr.shape[1]):
        if btype_arr[0][i] == 2 and btype_arr[1][i] == 2:
            bothunfolded += 1
        elif btype_arr[0][i] == 2 or btype_arr[1][i] == 2:
            oneunfolded += 1

    twounfolded_fraction = bothunfolded / btype_arr.shape[1]
    oneunfolded_fraction = oneunfolded / btype_arr.shape[1]
    noneunfolded_fraction = 1 - (twounfolded_fraction + oneunfolded_fraction)
    

    return b1_folded_fraction, b2_folded_fraction, oneunfolded_fraction,\
        noneunfolded_fraction, twounfolded_fraction, b1_meantype, b2_meantype, \
        b1_meanlength, b2_meanlength

def printDurationStats(btype_arr, blen_arr):
    # Mean time unfolded per bond
    timestep_no = btype_arr.shape[1]

    b1_unfolded = []
    counter = 0
    unfolded = False
    for i in range(timestep_no):
        if btype_arr[0][i] == 2 and unfolded:
            counter += 1
        elif btype_arr[0][i] == 1 and unfolded:
            unfolded = False
            b1_unfolded.append(counter * indextime)
            counter = 0
        elif btype_arr[0][i] == 2 and not unfolded:
            unfolded = True
        if i == timestep_no-1 and unfolded:
            b1_unfolded.append(counter * indextime)


    b2_unfolded = []
    counter = 0
    unfolded = False
    for i in range(timestep_no):
        if btype_arr[1][i] == 2 and unfolded==True:
            counter += 1
        elif btype_arr[1][i] == 1 and unfolded==True:
            unfolded = False
            b2_unfolded.append(counter * indextime)
            counter = 0
        if btype_arr[1][i] == 2 and unfolded==False:
            unfolded = True
        if i == timestep_no-1 and unfolded==True:
            b2_unfolded.append(counter * indextime)

    if b1_unfolded == []:
        b1_mean_unfolded_duration = 0
        b1_unfolded_std = 0
        b1_unfolded_se = 0
    else:
        b1_mean_unfolded_duration = np.mean(b1_unfolded)
        b1_unfolded_std = np.std(b1_unfolded)
        b1_unfolded_se = b1_unfolded_std / np.sqrt(len(b1_unfolded))

    if b2_unfolded == []:
        b2_mean_unfolded_duration = 0
        b2_unfolded_std = 0
        b2_unfolded_se = 0
    else:
        b2_mean_unfolded_duration = np.mean(b2_unfolded)
        b2_unfolded_std = np.std(b2_unfolded)
        b2_unfolded_se = b2_unfolded_std / np.sqrt(len(b2_unfolded))

    ############################################################################################
    # Mean time any bond is unfolded
    timestep_no = btype_arr.shape[1]

    b_unfolded = []
    counter = 0
    unfolded = False
    for i in range(timestep_no):
        if (btype_arr[0][i] == 2 or btype_arr[1][i] == 2) and unfolded:
            counter += 1
        elif (btype_arr[0][i] == 1 and btype_arr[1][i] == 1) and unfolded:
            unfolded = False
            b_unfolded.append(counter * indextime)
            counter = 0
        elif (btype_arr[0][i] == 2 or btype_arr[1][i] == 2) and not unfolded:
            unfolded = True
        if i == timestep_no-1 and unfolded:
            b_unfolded.append(counter * indextime)

    b_mean_unfolded_duration = np.mean(b_unfolded)
    b_unfolded_std = np.std(b_unfolded)
    b_unfolded_se = b_unfolded_std / np.sqrt(len(b_unfolded))

    ############################################################################################
    # Mean time one bond is unfolded
    timestep_no = btype_arr.shape[1]

    one_b_unfolded = []
    counter = 0
    oneunfolded = False
    for i in range(timestep_no):
        if btype_arr[0][i] == 2:
            if btype_arr[1][i] == 2 and oneunfolded:
                oneunfolded = False
                one_b_unfolded.append(counter * indextime)
            elif btype_arr[1][i] == 1 and not oneunfolded:
                oneunfolded = True
                counter = 0
            elif oneunfolded:
                counter += 1
        if btype_arr[1][i] == 2:
            if btype_arr[0][i] == 1 and not oneunfolded:
                oneunfolded = True
                counter = 0
            elif oneunfolded:
                counter += 1
        if i == timestep_no-1 and oneunfolded:
            one_b_unfolded.append(counter * indextime)

    one_b_mean_unfolded_duration = np.mean(one_b_unfolded)
    one_b_unfolded_std = np.std(one_b_unfolded)
    one_b_unfolded_se = one_b_unfolded_std / np.sqrt(len(one_b_unfolded))

    ############################################################################################
    # Equilibrium constant and free energy calculations
    # K_eq = (fraction of folded bs) / (fraction of unfolded bs)
    # Free energy G/kT = ln(K_eq). 
    # This is the difference in free energy between folded and unfolded states.

    if noneunfolded_fraction > 0:
        K_eq_UU = twounfolded_fraction / noneunfolded_fraction
        K_eq_FU = oneunfolded_fraction / noneunfolded_fraction
    else:
        K_eq_FU = 10e40
        K_eq_UU = 10e40

    G_FU = -np.log(K_eq_FU)
    G_UU = -np.log(K_eq_UU)

    mean_extension = np.mean(blen_arr[0]+blen_arr[1]-blen_arr[0][0]-blen_arr[1][0])

    #G_uncertainty = need multiple runs? YES

    return mean_extension, K_eq_UU, K_eq_FU, G_UU, G_FU, one_b_mean_unfolded_duration, \
        one_b_unfolded_std, one_b_unfolded_se, b1_mean_unfolded_duration, \
        b2_mean_unfolded_duration, b1_unfolded_std, b2_unfolded_std, \
        b1_unfolded_se, b2_unfolded_se, b_mean_unfolded_duration, b_unfolded_std, \
        b_unfolded_se

def plotgraphs(blen_arr, btype_arr, force, seed, mean_extension):
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

def printdata(b1_dict, b2_dict, molecule_dict, data_dict):
    print(f'______________|fraction of timesteps')
    print(f'both unfolded |{molecule_dict["UU fraction"]}')
    print(f'one unfolded  |{molecule_dict["FU fraction"]}')
    print(f'none unfolded  |{molecule_dict["FF fraction"]}')
    print()
    print(f'time of simulation: {(data_dict["length array"].shape[1]-1) * indextime * tau} ps')
    print(f'_______|mean type, folded fraction, mean length')
    print(f'bond 1 |{b1_dict["mean type"]}, {b1_dict["folded fraction"]}, {b1_dict["mean length"]}')
    print(f'bond 2 |{b2_dict["mean type"]}, {b2_dict["folded fraction"]}, {b2_dict["mean length"]}')
    print()
    print(f'_______|mean time unfolded (ns), std, se')
    print(f'bond 1 |{b1_dict["mean unfolded duration"]}, {b1_dict["unfolded std"]}, {b1_dict["unfolded se"]}')
    print(f'bond 2 |{b2_dict["mean unfolded duration"]}, {b2_dict["unfolded std"]}, {b2_dict["unfolded se"]}')
    print()
    print(f'_______|mean time one b unfolded (ns), std, se')
    print(f'bond   |{molecule_dict["FU mean duration"]}, {molecule_dict["FU duration std"]}, {molecule_dict["FU duration se"]}')
    print()
    print(f'_______|mean time any b unfolded (ns), std, se')
    print(f'bond   |{molecule_dict["mean unfold duration"]}, {molecule_dict["unfold std"]}, {molecule_dict["unfold se"]}')
    print()
    print(f'____________|equilibrium constant, free energy (kT)')
    print(f'Monomer FF  | 0')
    print(f'Monomer FU  | {molecule_dict["FU equilibrium constant"]}, {molecule_dict["FU free energy"]}')
    print(f'Monomer UU  | {molecule_dict["UU equilibrium constant"]}, {molecule_dict["UU free energy"]}')
    print()
    print(f'mean extension {molecule_dict["mean extension"]}')

btype_arr, blen_arr, force, seed = readfile()

b1_folded_fraction, b2_folded_fraction, oneunfolded_fraction,\
        noneunfolded_fraction, twounfolded_fraction, b1_meantype, b2_meantype, \
        b1_meanlength, b2_meanlength= printLengthStats(btype_arr, blen_arr)

mean_extension, K_eq_UU, K_eq_FU, G_UU, G_FU, one_b_mean_unfolded_duration, \
        one_b_unfolded_std, one_b_unfolded_se, b1_mean_unfolded_duration, \
        b2_mean_unfolded_duration, b1_unfolded_std, b2_unfolded_std, \
        b1_unfolded_se, b2_unfolded_se, b_mean_unfolded_duration, b_unfolded_std, \
        b_unfolded_se= printDurationStats(btype_arr, blen_arr)

b1_dict = {
    "mean type": b1_meantype,
    "mean length": b1_meanlength*sigma,
    "folded fraction": b1_folded_fraction,
    "mean unfolded duration": b1_mean_unfolded_duration*tau,
    "unfolded std": b1_unfolded_std*tau,
    "unfolded se": b1_unfolded_se*tau
}

b2_dict = {
    "mean type": b2_meantype,
    "mean length": b2_meanlength*sigma,
    "folded fraction": b2_folded_fraction,
    "mean unfolded duration": b2_mean_unfolded_duration*tau,
    "unfolded std": b2_unfolded_std*tau,
    "unfolded se": b2_unfolded_se*tau
}

molecule_dict = {
    "mean extension": mean_extension*sigma,
    "FU fraction": oneunfolded_fraction,
    "FU mean duration": one_b_mean_unfolded_duration*tau, 
    "FU duration std": one_b_unfolded_std*tau,
    "FU duration se": one_b_unfolded_se*tau,
    "FF fraction": noneunfolded_fraction,
    "UU fraction": twounfolded_fraction,
    "mean unfold duration": b_mean_unfolded_duration*tau,
    "unfold std": b_unfolded_std*tau,
    "unfold se": b_unfolded_se*tau,
    "FU equilibrium constant": K_eq_FU,
    "FU free energy": G_FU,
    "UU equilibrium constant": K_eq_UU,
    "UU free energy": G_UU
}

data_dict = {
    "type array": btype_arr,
    "length array": blen_arr*sigma

}

printdata(b1_dict, b2_dict, molecule_dict, data_dict)
plotgraphs(data_dict["length array"], data_dict["type array"], force, seed, molecule_dict['mean extension'])