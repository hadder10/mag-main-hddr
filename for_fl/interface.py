import streamlit as st
import pandas as pd
import time
import random

st.title("Подсистема защиты градиентов в FL")

# --- настройки ---
use_dp = st.checkbox("Включить защиту (DP)", value=True)
noise = st.slider("Noise multiplier", 0.0, 1.0, 0.3)
clip = st.slider("Clipping norm", 0.1, 5.0, 1.0)

rounds = st.slider("Количество раундов", 1, 20, 10)

# --- запуск ---
if st.button("Запустить обучение"):

    acc_list = []

    for r in range(rounds):
        time.sleep(0.3)

        # псевдо-метрики
        base_acc = 0.7 + r * 0.02
        if use_dp:
            base_acc -= noise * 0.05

        acc = min(base_acc, 0.9)
        acc_list.append(acc)

        st.write(f"Раунд {r+1}: accuracy = {acc:.3f}")

    # график
    df = pd.DataFrame({"accuracy": acc_list})
    st.line_chart(df)

    st.success("Обучение завершено")