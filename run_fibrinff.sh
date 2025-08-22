#!/bin/bash
# Set bond types to 2 and disable reactions

#=======================================
# USER SETTINGS

lmpinput="fibrin-monomer.lam"
stretch_force=0.0 # must be decimal values. 
#| xargs -n1 printf "%.1f\n" from chatgpt to make forces a decimal for all increments.
seed="54654651 54654653 54654659 54654661 54654685"
k_r_arr=$(seq 1 5 31 | xargs -n1 printf "%.1f\n")
k_theta_arr=$(seq 1 5 31 | xargs -n1 printf "%.1f\n")
k_r_prime=5.0

#=======================================

 
#=======================================
rm -rf output/K_rt_SeedFF
#=======================================


# Run simulations
for seedval in $seed ; do
  for k_r in $k_r_arr ; do
    for k_theta in $k_theta_arr ; do
      outdir="output/K_rt_SeedFF/k_r${k_r}_k_theta${k_theta}Seed${seedval}"
      mkdir -p "$outdir"
      ./lmp_mpi \
        -var stretch_force "$stretch_force" \
        -var seed "$seedval" \
        -var K_r_prime "$k_r_prime" \
        -var K_r "$k_r" \
        -var K_theta "$k_theta" \
        -var outdir "$outdir" \
        < "$lmpinput"
    done
  done
done