import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

qual = pd.read_csv("./data/quality.csv")
wine = pd.read_csv("./data/wine.csv")
hsb = pd.read_csv("./data/hsbdemo.csv")
crash = pd.read_csv("./data/crash.csv")

# print(qual.head(), "\n")
# print(wine.head())

## Log Regression Model - X: First Eight
qual['label'] = np.where(qual['label'] == 'B', 0, 1)
X = qual.drop(['S.No.','label'], axis=1)
X_normalized = X.apply(lambda x: (x-min(x))/max(x)-min(x))
y = qual['label']
# Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.2)


# logmod = LogisticRegression()
# logmod.fit(Xtrain, ytrain)

# predictions = logmod.predict(Xtest)

# print(accuracy_score(ytest, predictions))
# print(confusion_matrix(ytest, predictions))


################################# Do it 1000 times  #######################################################
# accuracy = [0] * 1000
# for i in range(1,1000):
#     Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized,y,test_size = 0.25)
#     logmod = LogisticRegression()
#     logmod.fit(Xtrain, ytrain)
#     # print(Xtest)
#     # print(ytest)
#     prediction = logmod.predict(Xtest)
#     # print(prediction)
#     acc = accuracy_score(ytest, prediction)
#     print(i)
#     accuracy[i] = acc


# plt.hist(accuracy, bins=10, facecolor='green', alpha = 0.75, label = 'All Features')

# ###################### Subset 1 #############################################################################
# X = qual[['num_words','num_misspelled','num_sentences']]
# X_normalized = X.apply(lambda x: (x-min(x))/max(x)-min(x))

# accuracy = [0] * 1000
# for i in range(1,1000):
#     Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized,y,test_size = 0.3)
#     logmod = LogisticRegression()
#     logmod.fit(Xtrain, ytrain)
#     # print(Xtest)
#     # print(ytest)
#     prediction = logmod.predict(Xtest)
#     # print(prediction)
#     acc = accuracy_score(ytest, prediction)
#     print(i)
#     accuracy[i] = acc

# # print(accuracy)
# plt.hist(accuracy, bins=10, facecolor='red', alpha = 0.75, label = 'Response Length & Quality')

# ###################### Subset 2 #############################################################################
# print(qual.columns)
# X = qual[['num_sentences', 'num_interrogative']]
# X_normalized = X.apply(lambda x: (x-min(x))/max(x)-min(x))

# accuracy = [0] * 1000
# for i in range(1,1000):
#     Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized,y,test_size = 0.3)
#     logmod = LogisticRegression()
#     logmod.fit(Xtrain, ytrain)
#     # print(Xtest)
#     # print(ytest)
#     prediction = logmod.predict(Xtest)
#     # print(prediction)
#     acc = accuracy_score(ytest, prediction)
#     print(i)
#     accuracy[i] = acc

# # print(accuracy)
# plt.hist(accuracy, bins=10, facecolor='blue', alpha = 0.75, label='Length and Interrogative')
# plt.xlabel("Accuracy")
# plt.ylabel("Count")
# plt.title("Accuracy of Logistic Regression Run 1000 Times")
# plt.legend()
# plt.savefig('LogReg_574.png')

############################################## Question 2 ############################################

####################################Feature Selection ######################################################
# print(wine.groupby('high_quality').min())
# print(wine.groupby('high_quality').mean())
# print(wine.groupby('high_quality').median())
# print(wine.groupby('high_quality').max())

# Compared the aggregate (min,max,avg,med) functions of each feature as well as their distributions broken down by quality in an effort to discriminate between the two; these were the values that were most different

# fig, (ax1,ax2) = plt.subplots(2)
# col = np.where(wine['high_quality'] == 1.0, 'blue','red')

# ax1.scatter(wine['volatile_acidity'],wine['alcohol'], c = col, alpha = 0.5, label = col)
# ax1.set(xlabel = ("Volatile Acidity"), ylabel = "Alcohol")
# ax2.scatter(wine['density'],wine['total_sulfur_dioxide'], c = col, alpha = 0.5, label = col)
# ax2.set(xlabel = "Density", ylabel = "Total Sulfur Dioxide")
# plt.suptitle("Distribution of Features")
# plt.savefig("FeatureDistribution.png")

###################################################### KNN 1x #############################################

# X = wine[['density','alcohol','volatile_acidity']]
# X_normalized = X.apply(lambda x: (x-min(x))/max(x)-min(x))
# y = wine[['high_quality']]

# Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized, y, test_size = 0.3)
# kmod = KNeighborsClassifier(n_neighbors=10).fit(Xtrain, ytrain)
# predictions = kmod.predict(Xtest)
# acc = accuracy_score(ytest, predictions)
# print(acc)
############################################# KNN 1000x for each K in [2,10] #############################################

# X = wine[['density','alcohol','volatile_acidity', 'total_sulfur_dioxide', 'citric_acid']]
# X_normalized = X.apply(lambda x: (x-min(x))/max(x)-min(x))
# y = wine[['high_quality']]


# accuracy = np.ones((100,9))

# for j in range(0, 9):
#     print("J: ", j)
#     print(pd.DataFrame(accuracy))
#     for i in range(0,100):
#         Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized,y,test_size = 0.3)
#         kmod = KNeighborsClassifier(n_neighbors=j+2)
#         kmod.fit(Xtrain, ytrain)
#         prediction = kmod.predict(Xtest)
#         # print(prediction)
#         acc = accuracy_score(ytest, prediction)
#         print(i)
#         accuracy[i,j] = acc

# acc = pd.DataFrame(accuracy)
# print(acc, "\n")
# print(acc.mean(axis = 0))

# k = [2,3,4,5,6,7,8,9,10]
# avg_acc = np.mean(accuracy, axis=0)

# print(avg_acc)
# plt.plot(k, avg_acc)
# plt.title("KNN Accuracy for Values K")
# plt.xlabel("Nearest Neighbors")
# plt.ylabel("Accuracy")
# plt.savefig("KNNAccuracy3.png")
# plt.show()

############################################ HSB KNN 1000x for each K in [2,10] #############################################



# X = hsb[['read','write','math', 'science']]
# X_normalized = X.apply(lambda x: (x-min(x))/max(x)-min(x))
# y = hsb[['prog']]


# accuracy = np.ones((100,9))

# for j in range(0, 9):
#     print("J: ", j)
#     print(pd.DataFrame(accuracy))
#     for i in range(0,100):
#         Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized,y,test_size = 0.3)
#         kmod = KNeighborsClassifier(n_neighbors=j+2)
#         kmod.fit(Xtrain, ytrain)
#         prediction = kmod.predict(Xtest)
#         # print(prediction)
#         acc = accuracy_score(ytest, prediction)
#         print(i)
#         accuracy[i,j] = acc

# acc = pd.DataFrame(accuracy)
# print(acc, "\n")
# print(acc.mean(axis = 0))

# k = [2,3,4,5,6,7,8,9,10]
# avg_acc = np.mean(accuracy, axis=0)

# print(avg_acc)
# plt.plot(k, avg_acc)
# plt.title("KNN Accuracy for Values K")
# plt.xlabel("Nearest Neighbors")
# plt.ylabel("Accuracy")
# plt.savefig("KNNAccuracyHSB.png")
# plt.show()

############################################ crash KNN 1000x for each K in [2,10] #############################################

X = crash[['Age','Speed']]
X_normalized = X.apply(lambda x: (x-min(x))/max(x)-min(x))
y = crash[['Survived']]


accuracy = np.ones((100,9))

for j in range(0, 9):
    print("J: ", j)
    print(pd.DataFrame(accuracy))
    for i in range(0,100):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X_normalized,y,test_size = 0.3)
        kmod = KNeighborsClassifier(n_neighbors=j+2)
        kmod.fit(Xtrain, ytrain)
        prediction = kmod.predict(Xtest)
        # print(prediction)
        acc = accuracy_score(ytest, prediction)
        print(i)
        accuracy[i,j] = acc

acc = pd.DataFrame(accuracy)
print(acc, "\n")
print(acc.mean(axis = 0))

k = [2,3,4,5,6,7,8,9,10]
avg_acc = np.mean(accuracy, axis=0)

print(avg_acc)
plt.plot(k, avg_acc)
plt.title("KNN Accuracy for Values K")
plt.xlabel("Nearest Neighbors")
plt.ylabel("Accuracy")
plt.savefig("KNNAccuracyCrash.png")
plt.show()