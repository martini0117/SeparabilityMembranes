#!/bin/sh

for dim in 5 10 15 20 25 30
do
    for var in 1 18 22 23 29 50 52 59 71 73 75 76 78 80 82 97 98 106 109 114 127 128 135 144 166 167 170 171 173 175 179 185 196 203 209
    do
        python3 ./test/SeparabilityProjectionSurf.py $var $dim
    done
done