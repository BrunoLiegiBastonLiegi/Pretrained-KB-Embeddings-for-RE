import argparse, json

parser = argparse.ArgumentParser(description='.')
parser.add_argument('-l',nargs='+')
args = parser.parse_args()

with open(args.l[0],'r') as f:
    a = json.load(f)

with open(args.l[1],'r') as f:
    b = json.load(f)

print(a,b)
