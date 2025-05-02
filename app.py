import streamlit as st

# Titel van de app
st.title("Mijn eerste Streamlit app")

# Invoer van de gebruiker
naam = st.text_input("Wat is je naam?")

# Als er een naam is ingevuld, geef een begroeting
if naam == "Ruveyda":
    st.success(f"Salam Alaikom habiba {naam}! Ik hou van jou! <3 xoxoxox")
else:
    print("Fy bitch")