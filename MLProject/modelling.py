
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import os

# Define the path to the preprocessed data (relative path for GitHub Actions)
preprocessed_data_path = 'preprocessed_diabetes.csv'

if __name__ == "__main__":
    print("Starting MLflow experiment...")

    # Enable MLflow autologging for scikit-learn
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # 1. Load the preprocessed data
        print(f"Loading preprocessed data from {preprocessed_data_path}...")
        df = pd.read_csv(preprocessed_data_path)
        print("Data loaded successfully. Head of DataFrame:")
        print(df.head())

        # 2. Separate features (X) and target (y)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        # 3. Split data into training and testing sets
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

        # 4. Initialize and train a Logistic Regression model
        print("Training Logistic Regression model...")
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X_train, y_train)
        print("Model training complete.")

        # MLflow autologging will automatically log metrics, parameters, and the model
        print("MLflow autologging will handle model evaluation and logging.")

        # Optional: Make predictions (for explicit logging if needed, though autologging covers it)
        y_pred = model.predict(X_test)
        # Note: Metrics are automatically logged by mlflow.sklearn.autolog() upon .fit()

        print("MLflow experiment concluded. Check MLflow UI for details.")
