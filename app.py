import streamlit as st

# Titel van de app
st.title("Mijn eerste Streamlit app")

# Invoer van de gebruiker
naam = st.text_input("Wat is je naam?")

# Als er een naam is ingevuld, geef een begroeting
if naam:
    st.success(f"Hallo {naam}! Welkom bij je eerste Streamlit app 🎉")
else:
    st.info("Vul je naam hierboven in om verder te gaan.")
