# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# from sklearn import tree
# import matplotlib.pyplot as plt
# import seaborn as sns  # Import seaborn for visualization

# from sklearn.metrics import confusion_matrix
# # Load the dataset
# data = pd.read_csv('dataset.csv')  # Replace with the path to your dataset

# print("the data set that i have: ", data.shape)


# # print the fist 5 rows data
# print(data.head())




# # Drop the 'ID' column as it is not a feature but an identifier
# features = data.drop(['ID', 'Status'], axis=1)
# # features = data.drop(['name', 'status'], axis=1)
# # labels = data['status']
# labels = data['Status']

# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(features , labels, test_size=0.2,random_state= 42 )

# # Create and train the Decision Tree classifier
# classifier = DecisionTreeClassifier(criterion='gini')
# classifier.fit(X_train, y_train)

# # Predict the labels for the test set
# predictions = classifier.predict(X_test)


# # Calculate and print the accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {100*accuracy:.2f}')


# # Generate and print the confusion matrix
# conf_matrix = confusion_matrix(y_test, predictions)
# print("Confusion Matrix:")
# print(conf_matrix)


# # # Optionally, visualize the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
#             xticklabels=['Healthy', 'Parkinson'], 
#             yticklabels=['Healthy', 'Parkinson'])
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()







# # Fit the model
# classifier.fit(X_train, y_train)



# # Optionally, visualize the tree
# fig = plt.figure(figsize=(14,15))
# _ = tree.plot_tree(classifier, 
#                    feature_names=data.columns[:-1],
#                    class_names=['Healthy', 'Parkinson'],
#                    filled=True)
# plt.show()

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn import tree
# import seaborn as sns

# # Load the dataset
# data = pd.read_csv('dataset.csv')

# print("The dataset I have: ", data.shape)
# print(data.head())

# # Prepare the features and labels
# features = data.drop(['ID', 'Status'], axis=1)
# labels = data['Status']

# # Verify and encode labels if necessary
# # This is a crucial step, ensure your labels are numeric for the ROC curve to work
# labels_encoded = pd.factorize(labels)[0]  # This encodes 'Healthy' as 0 and 'Parkinson' as 1

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# # Create and train the classifier
# classifier = DecisionTreeClassifier(criterion='gini')
# classifier.fit(X_train, y_train)

# # Predictions
# predictions = classifier.predict(X_test)

# # Accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {100*accuracy:.2f}')

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, predictions)
# print("Confusion Matrix:")
# print(conf_matrix)

# # Confusion matrix visualization
# # Uncomment to display
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Healthy', 'Parkinson'], yticklabels=['Healthy', 'Parkinson'])
# # plt.xlabel('Predicted labels')
# # plt.ylabel('True labels')
# # plt.title('Confusion Matrix')
# # plt.show()

# # ROC Curve
# probabilities = classifier.predict_proba(X_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, probabilities)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()

# # Decision Tree visualization
# # Uncomment to display
# # fig = plt.figure(figsize=(14, 15))
# # _ = tree.plot_tree(classifier, feature_names=data.columns[:-1], class_names=['Healthy', 'Parkinson'], filled=True)
# # plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset.csv')

print("The dataset I have: ", data.shape)
# # print the fist 5 rows data
print(data.head())

# Prepare the features and labels
features = data.drop(['ID', 'Status'], axis=1)
labels = pd.factorize(data['Status'])[0]  # Encode labels numerically

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,random_state= 42)

# Create and train the classifier
classifier = DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train, y_train)

# Predictions
predictions = classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {100*accuracy:.2f}')


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)

conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# ROC Curve
probabilities = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

# Display the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show(block=False)  # 'block=False' allows the code execution to continue and display all figures

# Display the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show(block=False)

# Display the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(classifier, feature_names=features.columns, class_names=['Healthy', 'Parkinson'], filled=True)
plt.title('Decision Tree')
plt.show(block=True)



