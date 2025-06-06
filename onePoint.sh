#!/bin/bash

# HOW TO USE:
# Put the onePoint.sh in the same folder as 
# OnePointFuncConsole.py and run the following command
# 		sbatch onePoint.sh $N $L $delta ${init conditions}
# where you replace $N, $L, $delta, and ${init conditions} 
# with your desired values of N, L, delta, and initial spots 
# for each up spin (values should be separated by spaces and 
# between 0 and L-1).

# You can also run this without sbatch to run it on your 
# computer (using onePoint.sh $N $L $delta ${init conditions})

# EXAMPLE COMMAND:
# 	sbatch onePoint.sh 2 5 0.03 2 3
# will generate a graph with N = 2, L = 5, and delta = 0.03.
# The up spins are located at positions 2 and 3. 

# The script will output a graph with the file name of 
# the form "n$N l$L d$D [${init conds}].png" where each 
# of the variables after the dollar signs are expanded to 
# the actual arguments supplied. 



# SBATCH COMMANDS 
# name of the job
#SBATCH -J generateOnePoint

#job resource specifications
#SBATCH -p share
#SBATCH --mem=5G
#SBATCH -c 4
#SBATCH --time=24:00:00

# files to write to
#SBATCH -o "plotStdOut (job %j).txt"
#SBATCH -e "plotStdError (job %j).txt"




# load python yay 
module load python

# creating a virtual display buffer (server doesn't have a 
# display, which pyplot needs, so we mimic one)
Xvfb :5 &
export DISPLAY=:5
XvfbPID=$(ps | grep Xvfb | grep -E -o ^[[:blank:]]*[[:digit:]]+ | grep -E -o [[:digit:]]+)

# check for python dependencies (matplotlib and scipy)
if [ "$(pip3 list | grep matplotlib)" = "" ]; then 
	echo "You don't have matplotlib installed yet"
	pip3 install --user matplotlib
fi 

if [ "$(pip3 list | grep scipy)" = "" ]; then 
	echo "You don't have scipy installed yet"
	pip3 install --user scipy
fi 

# run the thing 
python3 OnePointFuncConsole.py 

# delete virtual display buffer 
kill $XvfbPID
