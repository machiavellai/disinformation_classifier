# Disinformation Classifier

## Project Overview
The **Disinformation Classifier** is a machine-learning-based text classification system designed to distinguish between disinformation and factual statements. It utilizes **TF-IDF vectorization** and a **Multinomial Naïve Bayes classifier** to analyze textual data and predict its reliability. 


## Technologies Used
- **Python** (pandas, scikit-learn, NumPy)
- **Machine Learning** (Naïve Bayes Classifier)
- **Natural Language Processing** (TF-IDF Vectorization)
- **Data Cleaning and Preprocessing**

---
## Approach & Process


### **1. Data Collection & Preprocessing**
- The dataset was loaded from `data.csv`.
- Initial inspection revealed inconsistencies in column names and missing values.
- We implemented **data cleaning**, including:
  - Dropping NaN values
  - Filtering empty text entries
  - Ensuring the `Label` column was numeric.
- **Challenges Encountered:**
  - The dataset contained unexpected header misalignment.
  - Some rows had formatting errors that required manual fixes.


### **2. Exploratory Data Analysis (EDA)**
- We checked class distributions using:
  ```python
  df['Label'].value_counts()
  ```
- We ensured balanced training/testing sets using stratified sampling.
- **Challenges Encountered:**
  - The dataset was small, leading to imbalanced class distribution.


### **3. Model Selection & Training**
- We chose **Multinomial Naïve Bayes**, ideal for text classification tasks.
- The dataset was split into **80% training and 20% testing** using:
  ```python
  train_test_split(df['Text'], df['Label'], test_size=0.2, stratify=df['Label'])
  ```
- A pipeline with **TF-IDF Vectorization** was implemented.
- **Challenges Encountered:**
  - Initial cross-validation (`cv=5`) failed due to limited samples per class.
  - Solution: Adjusted `cv` dynamically based on class counts.

### **4. Model Evaluation & Testing**
- We evaluated the model using **cross-validation** and **test accuracy**.
- **Performance Metrics Used:**
  - Accuracy
  - Classification Report (Precision, Recall, F1-score)
- **Challenges Encountered:**
  - Skewed dataset resulted in high variance in model performance.

### **5. Error Analysis & Model Optimization**
- We analyzed misclassified texts to identify patterns.
- Improved preprocessing by filtering out noisy data.
- **Challenges Encountered:**
  - Some texts contained subtle misinformation, making classification difficult.

---
## Results & Findings
- **Final Model Accuracy:** 85% (Test Set)
- **Key Learnings:**
  - Ensuring high-quality data is crucial for accurate classification.
  - Class imbalance directly impacts cross-validation reliability.

---
## Challenges Faced & Solutions
| **Challenge** | **Solution Implemented** |
|--------------|----------------------|
| Dataset misalignment | Cleaned and reformatted columns |
| Missing values | Dropped NaN values and filtered empty text entries |
| Imbalanced dataset | Used stratified sampling and adjusted cross-validation |
| Cross-validation failure | Dynamically set `cv` based on minimum class count |
| Noisy and ambiguous text | Improved text preprocessing and feature extraction |

---
## Future Improvements
- Increase dataset size for better generalization.
- Experiment with deep learning (LSTMs, Transformers) for more advanced classification.
- Fine-tune hyperparameters to further improve performance.

---
## Conclusion
This project successfully built a working **disinformation classifier** using machine learning and NLP techniques. Despite challenges like dataset quality and model tuning, the final system achieved **promising accuracy**. Future iterations will focus on **expanding the dataset** and exploring **deep learning approaches**.

---
## Author
Developed by **Oghenerukewve**, a Full Stack Developer with expertise in Machine Learning and AI.

---
## Acknowledgments
Special thanks to the **scikit-learn** and **pandas** communities for their valuable documentation and resources.
