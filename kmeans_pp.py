import pandas as pd
import sys
import mykmeanssp as kmeans

def main():
    first_df = pd.read_csv("tests/input_1_db_1.txt",header=None)
    second_df = pd.read_csv("tests/input_1_db_2.txt",header=None)
    merged_df = pd.merge(first_df, second_df,on=0)
    points_array = merged_df.values.to_numpy()
    print(kmeans.fit(2, 2, 0.01, points_array.tolist())


if __name__ == "__main__":
    main()
