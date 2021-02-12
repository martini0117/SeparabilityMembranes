#!/bin/sh

for var in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20  #範囲の書き方(Bash独自) => {0..4}
do
    python3 ./kidney/InitNearestK.py $var
done

# for var in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20  #範囲の書き方(Bash独自) => {0..4}
# do
#     python3 ./test/CompareGtPre.py $var
# done

# for var in 15 16 17 18 19 20  #範囲の書き方(Bash独自) => {0..4}
# do
#     python3 ./test/CompareGtPre.py $var
# done
