# SalaryScope 💼
### Smart Employee Income Prediction Using Machine Learning

**Presented by:**  
Shaik Sheema Firdose  
Rajeev Gandhi Memorial College of Engineering and Technology (RGMCET)  
Department of Computer Science – AI & ML  
**AICTE Internship Student Registration ID:** STU64ec5408696ce1693209608

---

## 🚀 Project Overview

**SalaryScope** is a smart web application that predicts an employee’s income category using machine learning based on their demographic and professional data.  
In addition to the prediction, it provides **personalized career improvement suggestions** to help users upskill and enhance their earning potential.

---

## 🎯 Problem Statement

Many individuals are unaware of how their current qualifications and work patterns affect their income. SalaryScope addresses this gap by:
- Predicting whether a user's income is above or below ₹50,000.
- Recommending personalized actions to improve their career trajectory.

---

## 📊 Dataset Used

- **Dataset Source:** UCI Adult Income Dataset  
- **Attributes Include:** Age, Work Class, Education, Occupation, Hours per Week, Capital Gain/Loss, Marital Status, etc.
- **Preprocessing Steps:**
  - Missing value handling
  - Label encoding for categorical variables
  - Feature scaling and splitting for training/testing

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Libraries Used:**
  - `pandas`, `numpy`
  - `scikit-learn`
  - `joblib`
  - `streamlit`

---

## 🤖 Machine Learning Model

- **Model Used:** Logistic Regression (or your exact model, please confirm)
- **Performance:** Accuracy ~ (insert value if available)
- Model is saved as `model.pkl` and loaded during prediction.

---

## 🌟 Key Features

1. **Income Prediction:** Predicts if income is `<=50K` or `>50K` based on user input.
2. **Career Suggestions:** Personalized feedback based on:
   - **Education Level**
   - **Work Hours**
   - **Capital Gain/Loss**
   - **Marital Status**
   - **Occupation Type**
3. **Skill Resources:** Links to platforms like Coursera, YouTube, Simplilearn, Skill India Portal.
4. **Clean UI:** User-friendly interface with Streamlit.

---

## 📁 Repository Structure

```bash
├── app.py                   # Main application
├── model.pkl                # Trained ML model
├── suggestions.py           # Logic for personalized career tips
├── requirements.txt         # Python dependencies
├── README.md                # This file
```
How to Run the Project Locally
Clone the repository:


git clone https://github.com/Sheemafirdose/IBM_Ml_Project.git
cd IBM_Ml_Project
Install dependencies:


pip install -r requirements.txt
Run the Streamlit app:


streamlit run app.py
📌 Internship Acknowledgement
This project was developed as part of the AICTE Virtual Internship in Machine Learning in collaboration with IBM.
It reflects practical skills in real-world ML application design, model integration, and user-focused system development.


📬 Contact
For any queries or collaboration:

Email: sheemafirdose1311@gmail.com

GitHub: github.com/Sheemafirdose
