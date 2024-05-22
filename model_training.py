# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('Cleaned_SAML-D.csv')

# Assuming the target variable is 'Is_laundering' and other columns are features
X = df.drop('Is_laundering', axis=1)
y = df['Is_laundering']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=30, n_jobs=-1)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Function to evaluate the model using classification report and confusion matrix
def model_evaluation(model, X_test, y_test, color='Blues'):
    # Classification report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0), '\n')

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), cmap=color, annot=True, fmt='d')
    plt.xlabel('Predicted', size=12, weight='bold')
    plt.ylabel('Actual', size=12, weight='bold')
    plt.title('Confusion Matrix', weight='bold')
    plt.show()

# Evaluate the model using the function
model_evaluation(rf_model, X_test, y_test)

# Save the trained model to a file
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)