"""Merge dict and show statistics"""

from sys import argv

if len(argv) <= 1:
    print("Usage: xxx word_list1.txt word_list2.txt ...")
    exit()

merged_dict = set()
repeat = 0
for fname in argv[1:]:
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line in merged_dict:
                repeat += 1
            merged_dict.add(line)
for w in merged_dict:
    print(w)
print(len(merged_dict))
print(repeat)