#!/bin/sh

rm -r ./formatted
mkdir formatted
cd test_files
for filename in *.c; do
	echo "$filename"
	clang-format "$filename" > temp.c
	cat temp.c
	cat temp.c > ../formatted/"$filename"
	rm -f temp.c
done