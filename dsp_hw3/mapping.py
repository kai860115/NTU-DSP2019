import sys
import re

inFile = 'Big5-ZhuYin.map'
outFile = 'ZhuYin-Big5.map'

dictionary = {}

with open(inFile, 'r', encoding='big5hkscs') as fin:
    for line in fin:
        lineSplit = re.split(r'[/\s]', line)
        for s in lineSplit:
            if len(s) != 0:
                if s[0] not in dictionary:
                    dictionary[s[0]] = set()
                dictionary[s[0]].add(lineSplit[0])

with open(outFile, 'w', encoding='big5hkscs') as fout:
    for key, value in dictionary.items():
        line = key + '\t' + ' '.join(value) + '\n'
        fout.write(line)
