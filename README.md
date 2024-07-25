Project Team 6

**Project Title: Predicting Healthcare Employee Attrition with Machine Learning**

**Objective:**

This project leverages machine learning techniques to build a predictive model that identifies factors contributing to employee attrition (turnover) within the healthcare industry. By understanding the key drivers behind employees leaving, healthcare organizations can implement targeted strategies to improve retention and foster a more positive work environment.

**Project Structure:**

* **`Resources/`:**
    * **`watson_healthcare_modified.csv`:** The original dataset containing employee information.
* **`training/`:**
    * **`model.pkl`:** The trained AdaBoost classifier model.
    * **`scaler.pkl`:** The StandardScaler object used for data preprocessing.
    * **`ohe.pkl`:** The OneHotEncoder object used for encoding categorical variables.
    * **`choices.pkl`:** Dictionary mapping categorical variables to their possible values.
* **`app.py`:** Flask application for real-time attrition prediction.
* **`employee_data.db`:** SQLite database storing the employee dataset.
* **`README.md`:** This file.

**Key Variable:**

* **`Attrition`:** Binary variable (0 = Stayed, 1 = Left) indicating whether an employee left the organization. This is the target variable for our prediction models.

**Data Points:**
* **Employee Demographics:** Age, Gender, Marital Status, Education Level, Education Field, Distance From Home, etc.
* **Job-Related Factors:** Department, Job Role, Job Level, Job Satisfaction, Environment Satisfaction, Work-Life Balance, Years at Company, etc.

**Machine Learning Models:**

* **Random Forest Classifier:** A robust ensemble model known for its accuracy and ability to handle complex relationships in the data.
* **AdaBoost Classifier:**  Another ensemble model that combines multiple weak learners to create a strong predictor, often achieving high performance.

**Implementation:**

1. **Data Preprocessing:** 
    * The `watson_healthcare_modified.csv` dataset is cleaned and prepared in the Jupyter notebook (`training/model_training.ipynb`).
    * Categorical variables are encoded using one-hot encoding.
    * Numerical features are standardized for better model performance.

2. **Model Training:**
    * Two models are trained and evaluated: Random Forest and AdaBoost.
    * The model with the best performance (AdaBoost in this case) is saved as `model.pkl`.
    
3. **Prediction Interface:**
    * The `app.py` file creates a Flask web application.
    * It loads the trained model, scaler, and encoder.
    * Users can input employee data through a web form, and the app provides real-time attrition predictions.

**How to Run the Project:**

1. **Prerequisites:** 
   - Install Python and required libraries (`pip install pandas numpy matplotlib seaborn scikit-learn flask`).
2. **Set up Database:**
   - Run `app.py` once to create the `employee_data.db` SQLite database. 
3. **Run the App:** 
   - Execute `flask run` from your terminal.
   - Open your browser and navigate to `http://127.0.0.1:5000/`.
   - Fill out the form with employee details to get attrition predictions.
  
**Resources**
* Data Resource: https://www.kaggle.com/datasets/jpmiller/employee-attrition-for-healthcare/data
* Scaffolding for flask/html/app connection for a machine learning model: https://github.com/kevinclee26/sample_ml_app
* Ada Booster Classification Code: https://www.datacamp.com/tutorial/adaboost-classifier-python
