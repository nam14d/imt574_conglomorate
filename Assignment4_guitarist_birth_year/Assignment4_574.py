import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,plot_tree
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import matplotlib.pyplot as plt

blue = pd.read_csv("./data/blues_hand.csv")

########################## Question 1 ####################################################

#################################################### Regression ####################################################
# X = blue[['handPost','thumbSty']]
# y = blue[['brthYr']]


# reg_accuracy = np.ones((100))
# rfreg_accuracy = np.ones((100))

# reg_msee = np.ones((100))
# rfreg_msee = np.ones((100))

# for i in range(0,100):
#     Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.25)
#     reg = DecisionTreeRegressor()
#     rfreg = RandomForestRegressor(n_estimators=500)
#     reg.fit(Xtrain, ytrain)
#     rfreg.fit(Xtrain, ytrain)
#     reg_predictions = reg.predict(Xtest)
#     rfreg_predictions = rfreg.predict(Xtest)
#     # print(prediction)
#     reg_mse = mean_squared_error(ytest, reg_predictions)
#     rfreg_mse = mean_squared_error(ytest, rfreg_predictions)
#     print(i)
#     reg_msee[i] = reg_mse
#     rfreg_msee[i] = rfreg_mse


# plt.hist(reg_msee,bins = 30, color="red",alpha= 0.7, label = "DTree")
# plt.hist(rfreg_msee, bins = 30, color = "blue", alpha = 0.7, label = "RForest")
# plt.title("MSE Comparison Between Random Forest and Decision Tree")
# plt.xlabel("MSE")
# plt.ylabel("Count")
# plt.legend()
# plt.savefig("RFDT_MSE_DoubleFeatures_Comparison.png")
# plt.show()

# #################################################### Classification ####################################################

# blue[['brthYr']] = np.where(blue[['brthYr']] < 1901,0,np.where(blue[['brthYr']] < 1921,1,2))

# X = blue[['region','handPost','thumbSty']]
# y = blue[['brthYr']]

# for i in range(0,100):
#     Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.20)
#     reg = DecisionTreeClassifier()
#     rfreg = RandomForestClassifier(n_estimators=500)
#     reg.fit(Xtrain, ytrain)
#     rfreg.fit(Xtrain, ytrain)
#     reg_predictions = reg.predict(Xtest)
#     rfreg_predictions = rfreg.predict(Xtest)
#     # print(prediction)
#     reg_acc = accuracy_score(ytest, reg_predictions)
#     rfreg_acc = accuracy_score(ytest, rfreg_predictions)
#     print(i)
#     reg_accuracy[i] = reg_acc
#     rfreg_accuracy[i] = rfreg_acc


# plt.hist(reg_accuracy,bins = 10, facecolor="red",alpha= 0.7, label="DTree")
# plt.hist(rfreg_accuracy, bins = 10, facecolor = "blue", alpha = 0.7, label = "RForest")
# plt.title("Accuracy Comparison Between Random Forest and Decision Tree")
# plt.xlabel("Accuracy")
# plt.ylabel("Count")
# plt.legend()
# plt.savefig("RFDT_ACC_DoubleFeatures_Comparison.png")
# plt.show()



########################## Question 2 ####################################################
# X = blue[['region','handPost','thumbSty']]
# y = blue[['brthYr']]

# Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.25)

# reg = DecisionTreeRegressor()
# rfreg = RandomForestRegressor(n_estimators=100)

# reg.fit(Xtrain, ytrain)
# rfreg.fit(Xtrain, ytrain)

# reg_predictions = reg.predict(Xtest)
# rfreg_predictions = rfreg.predict(Xtest)

########################## Directly Compare Predicted Values ####################################################
# yprint = ytest.reset_index()[['brthYr']]

# reg_compare = pd.concat((yprint,pd.DataFrame(reg_predictions, columns=["DTree"]),pd.DataFrame(rfreg_predictions, columns=["RForest"])), axis = 1, join='outer')
# print(reg_compare,"\n")



# reg_mse = mean_squared_error(ytest, reg_predictions)
# rfreg_mse = mean_squared_error(ytest, rfreg_predictions)

# print(reg_mse, "\n")
# print(rfreg_mse)


############################## Compare RF and DT 100 Times As Regressors ############################################################
# reg_msee = np.ones((100))
# rfreg_msee = np.ones((100))

# for i in range(0,100):
#     Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.25)
#     reg = DecisionTreeRegressor()
#     rfreg = RandomForestRegressor(n_estimators=500)
#     reg.fit(Xtrain, ytrain)
#     rfreg.fit(Xtrain, ytrain)
#     reg_predictions = reg.predict(Xtest)
#     rfreg_predictions = rfreg.predict(Xtest)
#     # print(prediction)
#     reg_mse = mean_squared_error(ytest, reg_predictions)
#     rfreg_mse = mean_squared_error(ytest, rfreg_predictions)
#     print(i)
#     reg_msee[i] = reg_mse
#     rfreg_msee[i] = rfreg_mse


# plt.hist(reg_msee,bins = 30, color="red",alpha= 0.7, label = "DTree")
# plt.hist(rfreg_msee, bins = 30, color = "blue", alpha = 0.7, label = "RForest")
# plt.title("MSE Comparison Between Random Forest and Decision Tree")
# plt.xlabel("MSE")
# plt.ylabel("Count")
# plt.legend()
# plt.savefig("RFDT_MSE_Comparison.png")
# plt.show()



############################## Compare RF and DT As Classifiers ############################################################
# 1. Morph into Categorical Vars

blue[['brthYr']] = np.where(blue[['brthYr']] < 1901,0,np.where(blue[['brthYr']] < 1921,1,2))

X = blue[['region','handPost','thumbSty']]
y = blue[['brthYr']]

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.20)

reg = DecisionTreeClassifier()
# rfreg = RandomForestClassifier(n_estimators=100)

reg.fit(Xtrain, ytrain)
# rfreg.fit(Xtrain, ytrain)

reg_predictions = reg.predict(Xtest)
# rfreg_predictions = rfreg.predict(Xtest)

# reg_acc = accuracy_score(ytest, reg_predictions)
# rfreg_acc = accuracy_score(ytest, rfreg_predictions)

# print("Decision Tree Accuracy","\n",reg_acc,"\n")
# print("Random Forest Tree Accuracy","\n",rfreg_acc,"\n")

# reg_conf = confusion_matrix(ytest, reg_predictions)
# rfreg_conf = confusion_matrix(ytest, rfreg_predictions)

# print("Decision Tree Confusion Matrix","\n",reg_conf,"\n")
# print("Random Forest Confusion Matrix","\n",rfreg_conf,"\n")


############################## Compare RF and DT 100 Times As Classifiers ############################################################
# reg_accuracy = np.ones((100))
# rfreg_accuracy = np.ones((100))

# for i in range(0,100):
#     Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.20)
#     reg = DecisionTreeClassifier()
#     rfreg = RandomForestClassifier(n_estimators=500)
#     reg.fit(Xtrain, ytrain)
#     rfreg.fit(Xtrain, ytrain)
#     reg_predictions = reg.predict(Xtest)
#     rfreg_predictions = rfreg.predict(Xtest)
#     # print(prediction)
#     reg_acc = accuracy_score(ytest, reg_predictions)
#     rfreg_acc = accuracy_score(ytest, rfreg_predictions)
#     print(i)
#     reg_accuracy[i] = reg_acc
#     rfreg_accuracy[i] = rfreg_acc


# plt.hist(reg_accuracy,bins = 10, facecolor="red",alpha= 0.7, label="DTree")
# plt.hist(rfreg_accuracy, bins = 10, facecolor = "blue", alpha = 0.7, label = "RForest")
# plt.title("Accuracy Comparison Between Random Forest and Decision Tree")
# plt.xlabel("Accuracy")
# plt.ylabel("Count")
# plt.legend()
# plt.savefig("RFDT_ACC_Comparison.png")
# plt.show()

############################################################ Visualize Tree ############################################################ 

fig = plt.figure(figsize=(25,20))
_ = plot_tree(reg, feature_names=('region','handPost','thumbSty'),class_names=("Pre1900","Pre1920",'Post1920'), filled=True)
fig.savefig("decision_tree.png")