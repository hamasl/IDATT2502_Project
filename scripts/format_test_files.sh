#!/bin/sh

format_files() {
	mkdir ../formatted/"$1"
	echo "$1"
	for filename in $1/*.c; do
		clang-format "$filename" > ../formatted/"$filename"
		rm -f temp.c
	done
}

rm -r ./formatted
mkdir formatted
cd data/
for folder in CWE*; do
	format_files "$folder" &
done