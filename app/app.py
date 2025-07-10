import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page Config
st.set_page_config(page_title="💼 Employee Income Predictor", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        section[data-testid="stForm"] > div:first-child:empty {
            display: none !important;
        }

        .form-wrapper {
            border: 2px solid rgba(255,255,255,0.6);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            background-color: rgba(255,255,255,0.05);
            box-shadow: 0 0 10px rgba(255,255,255,0.2);
        }

        @media (prefers-color-scheme: light) {
            .form-wrapper {
                background-color: #f8f8f8;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
        }

        .stButton > button {
            padding: 0.6em 1.5em;
            font-size: 1.1em;
            font-weight: bold;
            color: inherit;
            background-color: transparent;
            border: 2px solid rgba(255,255,255,0.6);
            border-radius: 10px;
            box-shadow: 0 0 12px rgba(255,255,255,0.4);
            transition: all 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background: rgba(255,255,255,0.9);
            color: black;
        }

        .result-box {
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            background: #fff;
            color: #111;
            border: 2px solid #e0e0e0;
            box-shadow: 0 4px 24px 0 rgba(0,0,0,0.08);
            transition: background 0.3s, color 0.3s;
        }
        @media (prefers-color-scheme: dark) {
            .result-box {
                background: #181c24;
                color: #fff;
                border: 2px solid #444;
                box-shadow: 0 4px 24px 0 rgba(0,0,0,0.32);
            }
        }
        @media (prefers-color-scheme: light) {
            .result-box {
                background: #fff;
                color: #111;
                border: 2px solid #e0e0e0;
                box-shadow: 0 4px 24px 0 rgba(0,0,0,0.08);
            }
        }
        .highlight {
            color: #00e6ff;
        }

        h4 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- Heading ---
st.markdown("<h1 style='text-align:center;'>💼 Employee Income Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Predict whether a person's income exceeds 50K based on job and personal details.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

with st.form("income_form"):
    st.markdown("<h4>📝 Fill Employee Details</h4>", unsafe_allow_html=True)

    # Two vertical sections
    col1, col2 = st.columns(2)

    # --- Left Column ---
    with col1:
        st.markdown("#### 👤 Personal Info")
        age = st.selectbox("📅 Age", [""] + list(range(18, 76)))
        gender = st.radio("🧑 Gender", ['Male', 'Female'], horizontal=True)
        country = st.selectbox("🌐 Native Country", ["", 'United-States', 'India', 'Mexico', 'Philippines', 'Germany'])

        st.markdown("#### 🧑‍💼 Job Info")
        workclass = st.selectbox("🏢 Workclass", ["", 'Private', 'Self-emp-not-inc', 'Self-emp-inc',
                                                 'Federal-gov', 'Local-gov', 'State-gov'])
        occupation = st.selectbox("🧑‍💼 Occupation", ["", 'Exec-managerial', 'Craft-repair', 'Other-service',
                                                      'Sales', 'Prof-specialty', 'Handlers-cleaners'])
        marital = st.selectbox("💍 Marital Status", ["", 'Never-married', 'Married-civ-spouse', 'Divorced', 'Separated'])

    # --- Right Column ---
    with col2:
        st.markdown("#### 💵 Financial Info")
        edunum = st.slider("🎓 Education Number", 1, 16, value=10)
        hours_per_week = st.slider("⏱ Hours per Week", 1, 99, value=40)
        capital_gain = st.selectbox("📈 Capital Gain", ["", 0, 5000, 10000, 20000, 50000])
        capital_loss = st.selectbox("📉 Capital Loss", ["", 0, 1000, 2000, 3000])

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        submit = st.form_submit_button("🔍 Predict Income")

# --- Prediction ---
if submit:
    if "" in [age, capital_gain, capital_loss, workclass, marital, occupation, country]:
        st.warning("⚠ Please fill in all fields before predicting.")
    else:
        input_data = pd.DataFrame([{
            'age': int(age),
            'workclass': workclass,
            'educational-num': edunum,
            'marital-status': marital,
            'occupation': occupation,
            'gender': gender,
            'capital-gain': int(capital_gain),
            'capital-loss': int(capital_loss),
            'hours-per-week': hours_per_week,
            'native-country': country
        }])

        prediction = model.predict(input_data)[0]
        result = ">50K" if prediction == 1 else "<=50K"

        st.markdown(f"""
            <div class="result-box">
                💰 Predicted Income: <span class="highlight">{result}</span>
            </div>
        """, unsafe_allow_html=True)

        # ✅ Start of tips section
        st.markdown("### 🚀 Personalized Tips to Increase Income")

        tips = []

        if input_data["educational-num"].values[0] < 9:
            tip = "📚 Enhancing your education through diplomas or certifications can significantly improve your income potential."
            tips.append((tip, """
- [Coursera – Diploma Courses](https://www.coursera.org)
- [edX – Certificate Programs](https://www.edx.org)
- [Government Skill India Portal](https://www.skillindia.gov.in)
"""))

        elif 9 <= input_data["educational-num"].values[0] < 12:
            tip = "🎓 Consider upskilling with technical, managerial, or domain-specific courses to unlock better career opportunities."
            tips.append((tip, """
- [LinkedIn Learning – Upskilling](https://www.linkedin.com/learning)
- [Google Career Certificates](https://grow.google/certificates/)
- [YouTube – Skill Development Tips](https://www.youtube.com/results?search_query=career+upskilling)
"""))

        if input_data["hours-per-week"].values[0] < 35:
            tip = "⏱ Working more hours—if feasible—can improve visibility at work and boost earnings over time."
            tips.append((tip, """
- [Forbes – How Extra Hours Can Impact Your Career](https://www.forbes.com)
- [YouTube – Time Management Tips](https://www.youtube.com/results?search_query=time+management+for+career)
"""))

        if input_data["capital-gain"].values[0] == 0:
            tip = "💰 Start exploring passive income sources or investment strategies like mutual funds or freelancing for extra earnings."
            tips.append((tip, """
- [Zerodha Varsity – Investment Basics](https://zerodha.com/varsity/)
- [Groww – Mutual Funds Guide](https://groww.in/mutual-funds)
- [YouTube – Passive Income Ideas](https://www.youtube.com/results?search_query=passive+income+ideas)
"""))

        if input_data["capital-loss"].values[0] > 0:
            tip = "📉 Analyze past investments to avoid similar risks and adopt smarter financial planning for the future."
            tips.append((tip, """
- [ET Money – Investment Risk Management](https://www.etmoney.com/)
- [Moneycontrol – Portfolio Planning](https://www.moneycontrol.com/)
"""))

        if input_data["marital-status"].values[0] == "Never-married":
            tip = "💼 While personal choices vary, having strong support systems and networks can positively influence career development."
            tips.append((tip, """
- [LinkedIn Networking Guide](https://www.linkedin.com)
- [Coursera – Soft Skills for Career](https://www.coursera.org/learn/soft-skills)
- [YouTube – Building Career Networks](https://www.youtube.com/results?search_query=career+networking+skills)
"""))

        if input_data["occupation"].values[0] in ["Other-service", "Handlers-cleaners"]:
            tip = "🚀 Consider transitioning to roles in technology, finance, or healthcare by learning in-demand digital skills."
            tips.append((tip, """
- [FreeCodeCamp – Learn Programming](https://www.freecodecamp.org)
- [SkillUp by Simplilearn – Free IT Courses](https://www.simplilearn.com/skillup)
- [YouTube – Learn Python / Web Dev](https://www.youtube.com/results?search_query=learn+python+for+beginners)
"""))

        # ✅ Show each tip with "More Info"
        if not tips:
            st.success("👍 You're already making great progress. Keep building your skills and experience for continued growth!")
        else:
            for tip, resources in tips:
                st.info(tip)
                with st.expander("📎 More Info"):
                    st.markdown(resources)
