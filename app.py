
# app.py
import streamlit as st
from datetime import datetime
import random
from itertools import accumulate

st.set_page_config(page_title="Meu Site Bobo", page_icon="🪄", layout="centered")

st.title("🏡 Meu Site Bobo com Streamlit")
st.write("Bem-vindo! Um site simples pra testar e publicar.")

with st.sidebar:
    st.header("Menu")
    page = st.radio("Navegar para:", ["Home", "Dados", "Contato"])

if page == "Home":
    st.subheader("Home")
    st.image("https://picsum.photos/900/250", caption="Banner aleatório")
    nome = st.text_input("Seu nome")
    if nome:
        st.success(f"Olá, {nome}! ✨")

elif page == "Dados":
    st.subheader("Exemplo de dados (sem pandas)")
    n = st.slider("Quantidade de pontos", 5, 100, 20)
    passos = [random.uniform(-1, 1) for _ in range(n)]
    serie = list(accumulate(passos))  # caminhada aleatória
    st.line_chart(serie)
    st.caption("Gráfico de uma caminhada aleatória gerada na hora.")

elif page == "Contato":
    st.subheader("Contato (demo)")
    with st.form("form_contato"):
        email = st.text_input("Seu e-mail")
        msg = st.text_area("Mensagem")
        enviar = st.form_submit_button("Enviar")
    if enviar:
        st.info("Mensagem enviada! (apenas demonstração)")
        st.json({"email": email, "mensagem": msg, "quando": datetime.now().isoformat(timespec="seconds")})

st.caption("Feito com ❤️ usando Streamlit")
