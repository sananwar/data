import streamlit as st
from data import get_data

st.title("🎈 Simpele Streamlit App met Data")

# Data ophalen via data.py
df = get_data()

st.write("Hier is de data uit het andere bestand:")
st.dataframe(df)

# Extra interactie
stad = st.selectbox("Filter op stad:", df["Stad"].unique())

filtered_df = df[df["Stad"] == stad]
st.write("Geselecteerde resultaten:")
st.table(filtered_df)

# streamlit run app.py
