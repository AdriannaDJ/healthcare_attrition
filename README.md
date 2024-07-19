ML model

**Project Title: Analyzing Employee Attrition in the Healthcare System**

**Objective:**
This project aims to identify the key factors influencing employee attrition within the healthcare industry. By understanding the drivers behind employee turnover, we can develop targeted strategies to improve retention rates and create a more supportive work environment for healthcare professionals.

**Project Structure:**
* **`data/`:**
    * **`watson_healthcare_modified.csv`**
    * **`cleaned_data.csv`:** Data after cleaning and preprocessing.
* **`notebooks/`:**
    * **`01_data_exploration.ipynb`:**  Initial data exploration, visualization, and summary statistics.
    * **`02_data_preprocessing.ipynb`:**  Data cleaning, handling missing values.
    * **`03_model_training.ipynb`:** Building and training machine learning models to predict attrition.
    * **`04_model_evaluation.ipynb`:** Evaluating model performance, interpreting results, and identifying important features.
    * 
* **`README.md`:** This file.

**Key Variable:**

* **`Attrition`:**  Binary variable (0 or 1) indicating whether an employee has left the organization. This is our target variable for prediction.

**Data Points:**

* **Employee ID:** Unique identifier for each employee
* **Demographics:** Age, gender, MaritalStatus, Education, EducationField, DistanceFromHome, etc.
* **Work Characteristics:**
    * BusinessTravel
    * Department
    * EnvironmentSatisfaction
    * JobInvolvement
    * JobLevel
    * JobRole
    * JobSatisfaction
    * MonthlyIncome
    * OverTime
    * PercentSalaryHike
    * PerformanceRating
    * RelationshipSatisfaction
    * StandardHours
    * Shift


**Required Libraries:**

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn (or other preferred machine learning library)

**How to Run the Project:**

1. **Clone the Repository:**
   ```bash
   git clone <git@github.com:AdriannaDJ/healthcare_attrition.git>
   cd <healthcare_attrition>
   ```
2. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

**Expected Outcomes:**

* **Predictive Model:**  A machine learning model, capable of predicting employee attrition based on the provided data.
* **Feature Importance Analysis:** Identifying the most influential factors driving employee turnover in the healthcare system.
* **Insights and Recommendations:** Providing actionable insights and recommendations for healthcare organizations to improve employee retention.
