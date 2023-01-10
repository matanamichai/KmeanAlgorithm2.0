import pandas as pd
import sys
import numpy as np
#import mykmeanssp as kmeans

def main():
    first_df = pd.read_csv("tests/input_1_db_1.txt",header=None)
    second_df = pd.read_csv("tests/input_1_db_2.txt",header=None)
    merged_df = pd.merge(first_df, second_df,on=0)
    #print(merged_df)
    #points_array = merged_df.values.to_numpy()
    index = np.random.choice(merged_df.index,1)
    print(index)
    centroid_array = merged_df.iloc[index].to_numpy()
    print(centroid_array)
    merged_df_distance = merged_df.copy()
    while len(centroid_array)<5:
        merged_df_distance["distance"] = merged_df.apply(lambda row: min(np.linalg.norm(i - row) for i in centroid_array), axis=1) # 2
        merged_df_distance["ratio"] = merged_df_distance["distance"] / sum(merged_df_distance["distance"])  
        #print(merged_df)              # 3
        print("------")
        index = np.random.choice(merged_df.index,1, p=merged_df_distance["ratio"])
        centroid_array = np.append(centroid_array ,merged_df.iloc[index].to_numpy(),axis=0)
        print(centroid_array)

    
    #print(kmeans.fit(2, 2, 0.01, points_array.tolist()))


if __name__ == "__main__":
    main()
