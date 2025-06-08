import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("\U0001F4CA Voorspelling van Jeugdzorg- en WMO-voorzieningen per wijk (2025–2030)")

st.markdown("""
Welkom bij deze interactieve applicatie voor beleidsanalyse.

Deze tool voorspelt het verwachte gebruik van Jeugdzorg- en WMO-voorzieningen per wijk voor de jaren 2025 tot en met 2030.  
Door gebruik te maken van historische data en verwachte groei in huishoudens, krijg je inzicht in trends en toekomstige zorgbehoeften per wijk.

Scrol omlaag om de modelresultaten en interactieve grafieken te bekijken.
""")

try:
    excel_path = "Data onderzoek Anwar NIEUW.xlsx"
    xls = pd.ExcelFile(excel_path)
    ...
except Exception as e:
    st.error(f"Er ging iets mis bij het uitvoeren van de applicatie: {e}")

# Laad data
sheet_verrijking = pd.read_excel(xls, sheet_name='Verrijkte data')
sheet_jeugdzorg = pd.read_excel(xls, sheet_name='Jeugdzorg 2020-2024')
sheet_wmo = pd.read_excel(xls, sheet_name='WMO')

for df in [sheet_verrijking, sheet_jeugdzorg, sheet_wmo]:
    df.columns = df.columns.str.strip()

sheet_jeugdzorg['FAD'] = pd.to_datetime(sheet_jeugdzorg['FAD'], errors='coerce')
sheet_wmo['FAD'] = pd.to_datetime(sheet_wmo['FAD'], errors='coerce')
sheet_jeugdzorg['jaar'] = sheet_jeugdzorg['FAD'].dt.year
sheet_wmo['jaar'] = sheet_wmo['FAD'].dt.year

sheet_jeugdzorg = sheet_jeugdzorg.dropna(subset=['jaar']).copy()
sheet_wmo = sheet_wmo.dropna(subset=['jaar']).copy()
sheet_jeugdzorg['jaar'] = sheet_jeugdzorg['jaar'].astype(int)
sheet_wmo['jaar'] = sheet_wmo['jaar'].astype(int)

# === 2. Historische voorzieningen ===
jeugd = sheet_jeugdzorg.dropna(subset=['Laatst\nbekende\nwijk', 'Product\ncode', 'jaar'])
jeugd = jeugd.groupby(['Laatst\nbekende\nwijk', 'Product\ncode', 'jaar']).size().reset_index(name='Aantal_historisch')
jeugd = jeugd.rename(columns={'Laatst\nbekende\nwijk': 'wijk', 'Product\ncode': 'Voorziening'})
jeugd['Type_Voorziening'] = 'Jeugdzorg'

wmo = sheet_wmo.dropna(subset=['Laatst\nbekende\nwijk (CBS)', 'Segment', 'jaar'])
wmo = wmo.groupby(['Laatst\nbekende\nwijk (CBS)', 'Segment', 'jaar']).size().reset_index(name='Aantal_historisch')
wmo = wmo.rename(columns={'Laatst\nbekende\nwijk (CBS)': 'wijk', 'Segment': 'Voorziening'})
wmo['Type_Voorziening'] = 'WMO'

df_historical = pd.concat([jeugd, wmo])

combinations = df_historical[['wijk', 'Voorziening', 'Type_Voorziening']].drop_duplicates()
years = pd.DataFrame({'jaar': df_historical['jaar'].unique()})
full_combinations = combinations.merge(years, how='cross')
df_historical = pd.merge(full_combinations, df_historical,
                         on=['wijk', 'Voorziening', 'Type_Voorziening', 'jaar'], how='left')
df_historical['Aantal_historisch'] = df_historical['Aantal_historisch'].fillna(0)

# === 3. Dummy huishoudensdata ===
wijken = df_historical['wijk'].unique().tolist()
jaren = list(range(2020, 2031))
dummy = {'wijk': wijken}
for jaar in jaren:
    dummy[f'Aantal huishoudens ({jaar})'] = [1000 + i * 50 + (jaar - 2020) * 10 for i in range(len(wijken))]

df_huishoudens = pd.DataFrame(dummy)

# === 4. Merge + ML ===
huishoudens_melt = df_huishoudens.melt(id_vars='wijk', var_name='Jaar_str', value_name='Aantal huishoudens')
huishoudens_melt['jaar'] = huishoudens_melt['Jaar_str'].str.extract(r'\((\d{4})\)').astype(int)

ml_data = pd.merge(df_historical, huishoudens_melt[['wijk', 'jaar', 'Aantal huishoudens']],
                   on=['wijk', 'jaar'], how='left').fillna({'Aantal huishoudens': 0})

features = ['jaar', 'wijk', 'Voorziening', 'Type_Voorziening', 'Aantal huishoudens']
target = 'Aantal_historisch'
X_train, X_test, y_train, y_test = train_test_split(ml_data[features], ml_data[target], test_size=0.2, random_state=42)

cat_features = ['wijk', 'Voorziening', 'Type_Voorziening']
num_features = ['jaar', 'Aantal huishoudens']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', 'passthrough', num_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = np.maximum(0, np.round(model.predict(X_test)))

# === 5. Modelevaluatie ===
st.subheader("\U0001F4C8 Model Evaluatie")
st.columns(4)[0].metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
st.columns(4)[1].metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
st.columns(4)[2].metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
st.columns(4)[3].metric("R²", f"{r2_score(y_test, y_pred):.2f}")

# === 6. Toekomstvoorspellingen ===
toekomstige_jaren = range(2025, 2031)
combi = ml_data[['wijk', 'Voorziening', 'Type_Voorziening']].drop_duplicates()

future_rows = []
for jaar in toekomstige_jaren:
    for _, r in combi.iterrows():
        wijk = r['wijk']
        voorz = r['Voorziening']
        type_v = r['Type_Voorziening']
        huishoudens = df_huishoudens[df_huishoudens['wijk'] == wijk][f'Aantal huishoudens ({jaar})'].values[0]
        future_rows.append({
            'jaar': jaar,
            'wijk': wijk,
            'Voorziening': voorz,
            'Type_Voorziening': type_v,
            'Aantal huishoudens': huishoudens
        })

df_future = pd.DataFrame(future_rows)
df_future['Voorspeld Aantal ML'] = np.maximum(0, np.round(model.predict(df_future[features])))

eerste_jaar = min(toekomstige_jaren)
df_future['Voorspeld Aantal'] = df_future.apply(
    lambda r: np.round(r['Voorspeld Aantal ML'] * (1.04 ** (r['jaar'] - eerste_jaar))), axis=1
)

# === 7. Plotly Visualisatie ===
st.subheader("\U0001F4C9 Interactieve voorspellingen per wijk (2025–2030)")
df_plot = df_future.copy()
df_plot['Jaar'] = df_plot['jaar']

selected_wijk = st.selectbox("Selecteer een wijk:", sorted(df_plot['wijk'].unique()))
df_wijk = df_plot[df_plot['wijk'] == selected_wijk]

for type_v in ['Jeugdzorg', 'WMO']:
    df_v = df_wijk[df_wijk['Type_Voorziening'] == type_v]
    if df_v.empty:
        st.warning(f"Geen {type_v}-gegevens voor {selected_wijk}")
        continue

    top_voorzieningen = (
        df_v.groupby('Voorziening')['Voorspeld Aantal']
        .sum().nlargest(6).index.tolist()
    )
    df_v['Voorziening_Gegroepeerd'] = df_v['Voorziening'].apply(
        lambda x: x if x in top_voorzieningen else f"Overige {type_v}"
    )

    df_grouped = df_v.groupby(['Jaar', 'Voorziening_Gegroepeerd'])['Voorspeld Aantal'].sum().reset_index()

    fig = px.line(
        df_grouped,
        x='Jaar',
        y='Voorspeld Aantal',
        color='Voorziening_Gegroepeerd',
        markers=True,
        title=f"{type_v} in {selected_wijk}",
        labels={'Voorziening_Gegroepeerd': 'Voorziening'}
    )
    fig.update_layout(legend_title_text='Voorziening', height=500)
    st.plotly_chart(fig, use_container_width=True)