# Loan Risk Prediction System

## Overview

The **Loan Risk Prediction System** is a machine learning-based data analysis model developed to assess the risk associated with loan applicants. By analyzing financial history and various demographic features, the model categorizes applicants into three risk categories: **low**, **medium**, and **high**. This system is designed to help financial institutions predict the likelihood of loan default based on various factors such as income, credit score, loan amount, and more.

## Objective

The primary objective of this project is to develop a machine learning model that:
- Assesses loan applicants' risks based on their financial history.
- Classifies applicants into three categories: **Low**, **Medium**, **High**.
- Uses various features like **income**, **credit score**, and **loan amount** for prediction.
- Visualizes the prediction trends using Power BI.

## Features

- **Data Preprocessing**: Cleaning and transforming raw customer loan data for model training.
- **Risk Categorization**: Categorizing applicants into **low**, **medium**, or **high** risk using statistical and machine learning models.
- **Prediction Model**: Machine learning models to predict loan defaults based on various customer attributes.
- **Visualization**: Power BI reports to visualize trends and default predictions.

## Skills Utilized

- **Python** for data analysis and machine learning.
- **Power BI** for creating reports and visualizations.
- **SQL** for database management and query processing.
- **Machine Learning** for training predictive models.

---

## Dataset

The dataset contains the following columns:

| Column Name        | Description                                   |
|--------------------|-----------------------------------------------|
| `LoanID`           | Unique identifier for the loan                |
| `Age`              | Age of the loan applicant                     |
| `Income`           | Income of the applicant                       |
| `LoanAmount`       | Loan amount requested by the applicant        |
| `CreditScore`      | Credit score of the applicant                 |
| `MonthsEmployed`   | Number of months employed by the applicant    |
| `NumCreditLines`   | Number of credit lines held by the applicant  |
| `InterestRate`     | Interest rate of the loan                     |
| `LoanTerm`         | Term of the loan in months                    |
| `DTIRatio`         | Debt-to-Income Ratio                          |
| `Education`        | Education level of the applicant              |
| `EmploymentType`   | Employment status of the applicant            |
| `MaritalStatus`    | Marital status of the applicant               |
| `HasMortgage`      | Indicates if the applicant has a mortgage     |
| `HasDependents`    | Indicates if the applicant has dependents     |
| `LoanPurpose`      | Purpose of the loan                           |
| `HasCoSigner`      | Indicates if the applicant has a co-signer    |
| `Default`          | Whether the loan was defaulted (target column)|
| `Predicted_Default`| Predicted default status (for evaluation)     |

## Installation

### Prerequisites
Make sure you have the following libraries installed:
- **Python 3.x**
- **pandas** for data manipulation
- **scikit-learn** for machine learning models
- **matplotlib** and **seaborn** for data visualization
- **Power BI Desktop** for report generation

You can install the necessary Python libraries by running the following command:

```bash
pip install pandas scikit-learn matplotlib seaborn
