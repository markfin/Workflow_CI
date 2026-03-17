
# ML Model Training CI/CD Pipeline

## 📑 Project Overview
This repository contains an automated CI/CD pipeline for training machine learning models using MLflow. The pipeline automatically trains multiple models (Logistic Regression and Random Forest) on the Pima Indians Diabetes dataset and tracks all experiments using MLflow.

## 🚀 Features
- Automated model training with GitHub Actions
- MLflow experiment tracking
- Multiple model comparison (Logistic Regression vs Random Forest)
- Automatic dataset preprocessing
- Performance metrics logging

## 📊 Dataset
The Pima Indians Diabetes dataset contains medical data for 768 female patients. Features include:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target variable)

## 🛠️ How It Works

### CI/CD Pipeline
1. **Trigger**: Pipeline runs on push to main/master branch or manual trigger
2. **Environment Setup**: Python 3.9 with required dependencies
3. **Data Preparation**: Downloads and preprocesses the diabetes dataset
4. **Model Training**: Trains multiple models with MLflow tracking
5. **Artifact Management**: MLflow artifacts saved locally (excluded from git)

### MLflow Tracking
All experiments are tracked locally in the `mlruns/` directory including:
- Model parameters
- Performance metrics (accuracy, precision, recall, F1)
- Model artifacts
- Training metadata

## 📁 Repository Structure
