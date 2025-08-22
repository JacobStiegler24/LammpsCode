#!/bin/bash
# Set bond types to 2 and disable reactions

#=======================================
# USER SETTINGS

lmpinput="fibrin-monomer.lam"
stretch_force=0.0 # must be decimal values. 
#| xargs -n1 printf "%.1f\n" from chatgpt to make forces a decimal for all increments.
seed="54654651 54654653 54654659 54654661 54654685"
k_r_arr=$(seq 1 10 60 | xargs -n1 printf "%.1f\n")
k_theta=7.785
k_r=21.935

#=======================================

 
#=======================================
rm -rf output/K_rt_SeedUU
#=======================================


# Run simulations
for seedval in $seed ; do
  for k_r_prime in $k_r_arr ; do
      outdir="output/K_rt_SeedUU/k_r_prime${k_r_prime}_Seed${seedval}"
      mkdir -p "$outdir"
      ./lmp_mpi \
        -var stretch_force "$stretch_force" \
        -var seed "$seedval" \
        -var K_r_prime "$k_r_prime" \
        -var K_r "$k_r"\
        -var K_theta "$k_theta" \
        -var outdir "$outdir" \
        < "$lmpinput"
  done
done