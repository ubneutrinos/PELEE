#!/bin/bash

# Script for hadding together ntuple files based on a sam definition
# Author: C Thorpe (U of Manchester)
#
# Prestages files so best run inside nohup (or screen):
# Usage: nohup ./hadd_ntuple.sh <samdef> >& hadd_<samdef>.log &

cleanup=false
skip_prestage=true

########################################################################

def=$1
echo "Hadding files for sam def ${def}"

########################################################################
# Prestage the input def first

if [ $skip_prestage == false ]; then
    samweb prestage-dataset --defname=${def} --parallel=4 >& prestage_${def}.log 
    echo "Finished prestaging definition"
fi

########################################################################
# Make a fullpath list of the files

samweb list-file-locations --defname=${def} > file_locations_${def}.log
sed -i 's/\.root.*/\.root/g' file_locations_${def}.log
sed -i 's/enstore://g' file_locations_${def}.log
sed -i 's/\s\+/\//g' file_locations_${def}.log
grep -v '^dcache' file_locations_${def}.log > tmp_file_locations_${def}.log
mv tmp_file_locations_${def}.log file_locations_${def}.log
#sed -i 's/^dcache/d' file_locations_${def}.og

echo "Done making list of file locations"
 
########################################################################
# Convert the file list to xrootd format

while IFS= read -r line; do
  output=$(pnfsToXRootD "$line")
  echo "$output" >> "xrootd_locations_${def}.log"
done < "file_locations_${def}.log"

echo "Converted file locations to xrootd"

########################################################################
# Hadd together the xrood files

onelinefilelist=$(cat xrootd_locations_${def}.log | tr \\n ' ')
hadd -f ${def}.root ${onelinefilelist}

########################################################################
# Remove temp files/logs

if [ $cleanup == true ]; then
    rm file_locations_${def}.log
    rm xrootd_locations_${def}.log
    rm prestage_${def}.log 
fi

########################################################################

echo "Finished!"


