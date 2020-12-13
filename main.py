from a_utils import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# from a8_corr_svm_selftest import correction_svm_selftest
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt
import numpy as np
import collections
# Choose data
print("Choose data, 1 for old data (CRC), 2 for new data (IBD):")
dataset_choice = input("Choose (1 or 2) : ")
dataset_choice = int(dataset_choice)
name_predict = ""
data_name = ""
name_pre_new = ""
if (dataset_choice == 1):
    team_file = getOldDataset()
    data_name = "CRC"
elif (dataset_choice == 2):
    team_file = getNewDataset()
    data_name = "IBD"
#Choose predict method
predict_way = input("Choose predict method (1 is RF, 2 is SVM): ")
predict_way = int(predict_way)
if (predict_way == 1):
    clf_pre = RandomForestClassifier(n_estimators=1000, max_features='auto')
    name_predict = "RandomForest"
    name_pre_new = "RF"
elif (predict_way == 2):
    clf_pre = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    name_predict = "Support Vector Machine"
    name_pre_new = "SVM"
# Get data in file
data = pd.read_csv(team_file.train)
colName = data.columns
df = pd.DataFrame(data, columns=colName)
df.head()
X = df[colName]
y = df[team_file.resultColName]
columns = X.columns.tolist()
X_new = X.drop(X.columns[[0, len(columns)-1]], axis=1)
data_top = X_new.columns.values
# Selection feature
clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
clf.fit(X_new, y)
name_file = []
sorted_idx = (-clf.feature_importances_).argsort(-1)[:35]
features_value = clf.feature_importances_[sorted_idx]
new_name = data_top[sorted_idx]
list_new = []
nTimes = 100
acc_if = 0.0
print("Bắt đầu kết quả ----------------- ")
acc_average = []
acc_aver_total = []
mul_list_dict = collections.defaultdict(list)
list_new = []
for x in range(len(team_file.listFileTest)):
    name_file.append(team_file.listFileTest[x])
    average=0
    for n in range(nTimes):
        print("Chạy test trên " + team_file.listFileTest[x] + ". Chạy lặp " + str(n+1) + " feature.")
        if nTimes == 0:
            break
        data_yu = pd.read_csv(team_file.listFileTest[x])
        importanceFeature = new_name[:(n+1)]
        X_train_IF = df[importanceFeature]
        y_train_IF = y
        df_IF = pd.DataFrame(data_yu, columns=importanceFeature).fillna(0)
        X_Test_IF = df_IF[importanceFeature]
        y_Test_IF = data_yu[team_file.resultColName]

        clf_pre.fit(X_train_IF, y_train_IF)  # Build a forest of trees from the training set (X, y).
        y_Predict_IF = clf_pre.predict(X_Test_IF)

        acc_if += metrics.accuracy_score(y_Test_IF, y_Predict_IF)
        mul_list_dict[(x + 1)].append(round(acc_if, 4))
        print(round(acc_if, 4))
        acc_average.append(round(acc_if, 4))
        if nTimes == 0:
            break
        acc_if = 0.0
    acc_if = 0.0
    average = sum(acc_average)/len(acc_average)
    print("Average ACC of "+team_file.listFileTest[x]+ " :")
    print(average)
    acc_average = []
name_columns_graph = []
name_file.append("Average ACC")
# get name column
for x in range(nTimes):
    name_columns_graph.append(str(x+1))
list_acc_file = []
# Append result into one list
for x in range(len(team_file.listFileTest)):
    acc_file = np.array(mul_list_dict[(x+1)])
    list_acc_file.append(acc_file)
data_new = np.array(list_acc_file)
# data for draw boxplot
df_new = pd.DataFrame(data = data_new, columns = name_columns_graph)
# Calculate average ACC of each a couple feature
av_row = data_new.mean(axis=0)
# append average value and result
data_export = np.vstack([data_new, av_row])
# data for export csv
df1 = pd.DataFrame(list(data_export),
                   index=name_file,
                   columns=name_columns_graph)
name_result = data_name + "_" + name_pre_new
df1.to_csv('result_'+name_result+'.csv')
# data for drawing line
data_line = list(av_row)
# Code draw line
plt.plot(name_columns_graph, data_line, color='red', marker='o')
plt.xticks(rotation=90)
plt.title('Predict using ' + name_predict, fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(False)
# Code draw Boxplot
df_new.plot.box(grid=False, rot=90)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

plt.show()


