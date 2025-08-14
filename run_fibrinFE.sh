#!/bin/bash


#=======================================
# USER SETTINGS

lmpinput="fibrin-monomer-FE.lam"
seed="54654651 54654653 54654659 54654661 54654685"
outdir="output"

k_r=21.935
k_r_prime=21.935
k_theta=7.785

#=======================================

rm -rf output/ForceExtension

# Run simulations
for seedval in $seed ; do
  outdir="output/ForceExtension/Force-ExtensionSeed${seedval}"
  mkdir -p "$outdir"
  ./lmp_mpi \
    -var seed "$seedval" \
    -var outdir "$outdir" \
    -var K_r "$k_r" \
    -var K_r_prime "$k_r_prime" \
    -var K_theta "$k_theta" \
    < "$lmpinput"

done
