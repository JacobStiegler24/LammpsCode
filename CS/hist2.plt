# histogram.gnuplot
set terminal pngcairo enhanced
set output 'hist2.png'


FNAME1='bond_length.out'


# Set the bin width
bin_width = 0.005

# Count total number of data points (read manually or with stats)
stats FNAME1 nooutput
total1 = STATS_records


# Normalize by total and bin width
norm1 = 1.0 / (total1 * bin_width)

M_PI=8

set xrange [0:3.14]

# Define a binning function
bin(x, width) = width * floor(x / width)

# Set the style
set style fill empty # 0.5
set boxwidth bin_width

# Set labels
set xlabel "Value"
set ylabel "Normalised Frequency"
set title "Histogram of Data"

K_t=2.3
k_r=K_t*2

R0=1.0
Rmean = 3.14

STD=sqrt(1.0/(k_r))

set key right top 
# Plot
plot \
FNAME1 using (bin(acos($3), bin_width)):(norm1) smooth freq with p  ps 2 pt 7 lc rgb 'red' title 'initial',\
5.5*sin(x)*1.0/sqrt(2*3.14*STD**2)*exp(-(x-Rmean)**2/(2*STD**2)) w l lw 3 lc rgb 'black' title 'theory'
