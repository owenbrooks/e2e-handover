import pandas as pd

df= pd.read_csv('data/2021-12-09-04:56:05.csv', sep=' ')
# print(df)
# print(df.describe())
print(df["gripper_is_open"].value_counts())