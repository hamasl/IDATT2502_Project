#!/bin/sh
rm -r ./formatted
mkdir formatted
cd data/
for folder in CWE*; do
	mkdir ../formatted/"$folder"
	echo "$folder"
	for filename in $folder/*.c; do
		clang-format "$filename" > ../formatted/"$filename"
		rm -f temp.c
	done
done