import pandas as pd

TreePd = pd.read_csv("C:\\Users\\Hassan\\Desktop\\ML\\decision_tree\\1st Segment.csv" , on_bad_lines='skip')
xTrain = TreePd["Travel_Time"]
yTrain = TreePd.drop(["Travel_Time"], axis=1)


print(xTrain)
print(yTrain)