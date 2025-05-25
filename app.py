import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

# --- Función para cargar datos ---
#@st.cache(allow_output_mutation=True)
def load_data(path: str = 'gastos_domingo_porcentaje.xlsx'):
    return pd.read_excel(path)

# --- Función para entrenar el modelo ---
#@st.cache(allow_output_mutation=True)
def train_model(df: pd.DataFrame):
    features = ['ubicacion_desayuno', 'almuerza', 'costo_gasolina', 'costo_pasaje', 'gastos_otros']
    X = df[features]
    y = df['porcentaje_gasto_universidad']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['costo_gasolina', 'costo_pasaje', 'gastos_otros']),
        ('cat', OneHotEncoder(drop='if_binary'), ['ubicacion_desayuno', 'almuerza']),
    ])

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('reg',  LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    cv_mae = -cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error').mean()

    return pipeline, mae, rmse, r2, cv_mae
st.title('Presupuesto y % de gasto en la universidad')
df = load_data()
st.dataframe(df.head())

pipeline, mae, rmse, r2, cv_mae = train_model(df)

st.metric('MAE', f"{mae:.2f} %")
st.metric('RMSE', f"{rmse:.2f} %")
st.metric('R²', f"{r2:.3f}")
st.write(f"MAE CV 5-fold: {cv_mae:.2f} %")

st.write('---')
st.header('Presupuesto individual')
with st.form('budget_form'):
    ubicacion = st.selectbox('Dónde desayuna', ['casa','universidad'])
    almuerza = st.checkbox('¿Almuerza?', True)
    cd = st.number_input('Costo desayuno (Q)', 0.0)
    ca = st.number_input('Costo almuerzo (Q)', 0.0)
    cg = st.number_input('Costo gasolina (Q)', 0.0)
    cp = st.number_input('Costo pasaje (Q)', 0.0)
    go = st.number_input('Otros gastos (Q)', 0.0)
    submit = st.form_submit_button('Calcular')

if submit:
    total = cd+ca+cg+cp+go
    uni   = cd+ca
    pct   = pipeline.predict(pd.DataFrame([{ 'ubicacion_desayuno': ubicacion, 'almuerza': int(almuerza), 'costo_gasolina': cg, 'costo_pasaje': cp, 'gastos_otros': go }]))[0]
    st.write(f"Gasto total domingo: Q{total:.2f}")
    st.write(f"Gasto en universidad: Q{uni:.2f}")
    st.success(f"Porcentaje en universidad: {pct:.2f}%")