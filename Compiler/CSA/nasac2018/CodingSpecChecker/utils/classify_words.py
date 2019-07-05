from nltk.corpus import wordnet
from sys import argv

if len(argv) != 7:
    print("Usage: xxx dict_path verb_path noun_path adj_path adj_sat_path adv_path")
    exit()

verbs = set()
nouns = set()
adjs = set()
adj_sats = set()
adv = set()

with open(argv[1]) as dict_file:
    for line in dict_file:
        line = line.strip()
        for s in wordnet.synsets(line):
            if (s.pos() == 'n'):
                nouns.add(line)
            elif (s.pos() == 'v'):
                verbs.add(line)
            elif (s.pos() == 'a'):
                adjs.add(line)
            elif (s.pos() == 's'):
                adj_sats.add(line)
            elif (s.pos() == 'r'):
                adv.add(line)

def output(word_set, outputfile_name):
    with open(outputfile_name, 'w+') as f:
        for w in word_set:
            print(w, file=f)

output(verbs, argv[2])
output(nouns, argv[3])
output(adjs, argv[4])
output(adj_sats, argv[5])
output(adv, argv[6])
