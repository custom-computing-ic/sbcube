import argparse 
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument("--load_csv", default=None, type=str, help="dir of csv")

args = parser.parse_args()

data_csv = pd.read_csv(args.load_csv)
print(data_csv)
