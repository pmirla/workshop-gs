

```python

import pandas as pd
import numpy as np

# Your original dataset
data = {
    "Class": ["C1", "C2", "C2", "C1", "C1", "C1", "C1", "C1", "C2", "C1", "C1", "C1",
              "C2", "C2", "C2", "C2", "C2", "C2", "C2", "C2", "C2", "C2", "C2", "C2",
              "C2", "C2", "C2", "C2", "C1", "C2", "C2", "C2"],
    "x": [0.370278, 0.413011, 0.209686, 0.304381, 0.187805, 0.478294, 0.542574, 0.571402,
          0.28356, 0.165391, 0.429445, 0.775779, 0.798157, 0.601222, 0.830028, 0.695036,
          0.32777, 0.53036, 0.569237, 0.442623, 0.777587, 0.792177, 0.884909, 0.60669,
          0.941629, 0.910623, 0.895706, 0.962266, 0.951353, 0.754484, 0.563402, 0.867663],
    "y": [0.443383, 0.554997, 0.469474, 0.750514, 0.689745, 0.658589, 0.638511, 0.492332,
          0.598919, 0.244694, 0.342199, 0.587151, 0.289823, 0.262215, 0.371523, 0.251658,
          0.115146, 0.181018, 0.216375, 0.164681, 0.410796, 0.213162, 0.119249, 0.113607,
          0.667047, 0.80755, 0.446979, 0.35717, 0.326453, 0.483104, 0.417541, 0.507274]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Replace 'C1' with 0 and 'C2' with 1 in the Class column
df['Class'] = df['Class'].map({'C1': 0, 'C2': 1})

# Introduce some overlap by adding random noise to a subset of each class
np.random.seed(42)  # For reproducibility
noise_intensity = 0.05
indexes_to_modify = np.random.choice(df.index[df['Class'] == 0], size=int(len(df) * 0.1), replace=False)
df.loc[indexes_to_modify, 'x'] += noise_intensity * np.random.randn(len(indexes_to_modify))
df.loc[indexes_to_modify, 'y'] += noise_intensity * np.random.randn(len(indexes_to_modify))

indexes_to_modify = np.random.choice(df.index[df['Class'] == 1], size=int(len(df) * 0.1), replace=False)
df.loc[indexes_to_modify, 'x'] += noise_intensity * np.random.randn(len(indexes_to_modify))
df.loc[indexes_to_modify, 'y'] += noise_intensity * np.random.randn(len(indexes_to_modify))

df.head()



```

![image](https://github.com/pmirla/workshop-genus/assets/5429924/0ab0e267-712a-44be-86b8-ed7303ae6910)



```python

# Assuming 'df' is the DataFrame created from the dataset provided earlier

import matplotlib.pyplot as plt

# Plotting the scatter plot with colors by class
plt.figure(figsize=(8, 6))
plt.scatter(df[df['Class'] == 0]['x'], df[df['Class'] == 0]['y'], c='green', label='Class 0')
plt.scatter(df[df['Class'] == 1]['x'], df[df['Class'] == 1]['y'], c='blue', label='Class 1')
plt.title('Scatter Plot by Class')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()




```






```python

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report, roc_auc_score

# Assuming 'df' is your DataFrame and it has the feature columns named 'x' and 'y'
# and the target column named 'Class'

# 1. Split the DataFrame into features and target
X = df[['x', 'y']]
y = df['Class']

# 2. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Fit a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict the target on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 5. Calculate accuracy, precision, recall, F1-score, and ROC AUC
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
report = classification_report(y_test, y_pred)

# 6. Compute the log loss for the predictions
loss = log_loss(y_test, y_pred_proba)

# Print the metrics
print(f'Accuracy: {accuracy}')
print(f'ROC AUC: {roc_auc}')
print(report)
print(f'Log Loss: {loss}')



```



```python

import matplotlib.pyplot as plt
import numpy as np

# Create a mesh grid
x_min, x_max = X['x'].min() - 1, X['x'].max() + 1
y_min, y_max = X['y'].min() - 1, X['y'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict class for each point on the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X['x'], X['y'], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

# Define plot labels and titles
plt.xlabel('x')
plt.ylabel('y')
plt.title('Logistic Regression Decision Boundary')
plt.show()



```

![image](https://github.com/pmirla/workshop-genus/assets/5429924/746858a8-6d11-438b-84fc-302f9f114fbc)



```pyhton
# Assuming 'model' is your trained logistic regression model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

```

```bash

Accuracy: 0.8
ROC AUC: 0.625
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         2
           1       0.80      1.00      0.89         8

    accuracy                           0.80        10
   macro avg       0.40      0.50      0.44        10
weighted avg       0.64      0.80      0.71        10

Log Loss: 0.5656339944811452


```


```pyhton
import matplotlib.pyplot as plt
import numpy as np

# Assume 'model' is your pre-trained logistic regression model
# X and y are your features and labels for the entire dataset
# X_test and y_test are your features and labels for the test set

# Create a mesh grid for plotting the decision boundary
x_min, x_max = X['x'].min() - 1, X['x'].max() + 1
y_min, y_max = X['y'].min() - 1, X['y'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict class for each point on the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)

# Scatter plot for the test data only
plt.scatter(X_test['x'], X_test['y'], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k')

# Define plot labels and titles
plt.xlabel('x')
plt.ylabel('y')
plt.title('Logistic Regression Decision Boundary (Test Set Only)')
plt.show()


```



![image](https://github.com/pmirla/workshop-genus/assets/5429924/25c459e9-7428-44f4-bcf0-efb8e8cc1cfc)




TP: The model correctly predicts the positive class.

TN: The model correctly predicts the negative class.

FP: The model incorrectly predicts the positive class (a type I error).

FN: The model incorrectly predicts the negative class (a type II error).


Precision = TP / (TP + FP)

Recall = TP / (TP + FN)




