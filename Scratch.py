import argparse
import os

P = argparse.ArgumentParser()
P.add_argument("--path")
args = P.parse_args()
# args.path = os.path.abspath(args.path)
print(args)