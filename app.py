import streamlit as st
import pandas as pd
import joblib

# === Загружаем bundle с моделью и энкодерами ===
bundle = joblib.load("car_model_bundle.pkl")
model = bundle["model"]
le_gender = bundle["encoder_gender"]
le_color = bundle["encoder_color"]

# === Заголовок ===
st.title("🚗 Car Model Predictor")
st.write("Выбери характеристики и узнай, какую модель предскажет твой RandomForest!")

# === Ввод пользователя ===
gender = st.selectbox("Gender", le_gender.classes_)
color = st.selectbox("Color", le_color.classes_)

# === Кнопка предсказания ===
if st.button("🔮 Predict"):
    df_input = pd.DataFrame([{
        "Gender": gender,
        "Color": color
    }])
    
    # Преобразуем строки в числовые коды
    df_input["Gender"] = le_gender.transform(df_input["Gender"])
    df_input["Color"]  = le_color.transform(df_input["Color"])
    
    # Предсказываем
    pred = model.predict(df_input)
    st.success(f"Predicted car model: **{pred[0]}**")
