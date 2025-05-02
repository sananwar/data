import streamlit as st

# Titel van de app
st.title("Douchen met Sex")

# Invoer van de gebruiker
naam = st.text_input("Wil je dat ik ga douchen?")

# Als er een naam is ingevuld, geef een begroeting
if naam == "Ja":
    st.success(f"Isgoed, kom naar de badkamer. Dan kunnen we samen douchen")
else:
    st.info("Dit is je kans.")
