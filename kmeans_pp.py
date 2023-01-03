import pandas as pd
import sys

def main():
    first_df = pd.read_csv("tests/input_1_db_1.txt",header=None)
    second_df = pd.read_csv("tests/input_1_db_2.txt",header=None)
    print(first_df.head(4))
    # first_df.columns = [lambda : len(first_df.columns)]
    print("-------")
    print(second_df.head(4))
    print("-------")
    merged_df = pd.merge(first_df, second_df,on=0)
    print(merged_df.head(4))
    listOfPoint = merged_df.values.tolist()
    print(listOfPoint)


if __name__ == "__main__":
    main()
