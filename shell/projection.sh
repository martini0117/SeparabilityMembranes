#!/bin/sh

for var in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20  #範囲の書き方(Bash独自) => {0..4}
do
    for dim in 35 40 45 50 55 60 65 70 75 80
    do
        python3 ./kidney/MakeProjectionMatrix.py $var $dim
    done
done
