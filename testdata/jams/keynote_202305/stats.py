import os

for filename in os.listdir('.'):
    if not filename.startswith('p'):
        continue
    with open(filename) as f:
        words = 0
        for line in f:
            words += len(line.strip())
        print(filename, words)