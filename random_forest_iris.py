import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Random Forest Model Example")
print("-" * 40)

# load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create model
model = RandomForestClassifier(n_estimators=100)

# train model
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# check accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# show feature importance
print("\nFeature Importance")
for name, value in zip(iris.feature_names, model.feature_importances_):
    print(name, ":", round(value, 3))

# test with a new flower sample
print("\nTesting with a new sample")

sample = np.array([[5.1, 3.5, 1.4, 0.2]])

result = model.predict(sample)
prob = model.predict_proba(sample)

print("Sample values:", sample[0])
print("Predicted flower:", iris.target_names[result[0]])
print("Confidence:", round(prob[0][result[0]] * 100, 2), "%")