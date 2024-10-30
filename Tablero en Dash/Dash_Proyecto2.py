import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Cargar la base de datos
file_path = 'Bank.csv'
df = pd.read_csv(file_path, delimiter=';')

# Asegurar que la columna 'month' esté en minúsculas y sin valores nulos
df['month'] = df['month'].str.lower()
df = df.dropna(subset=['month'])
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Cargar y preparar los datos
file_path = 'Bank.csv'
bank_data = pd.read_csv(file_path, delimiter=';')
bank_data.columns = bank_data.columns.str.strip().str.lower()

# Manejo de valores nulos y codificación de variables
bank_data.fillna('unknown', inplace=True)

le_job = LabelEncoder()
le_marital = LabelEncoder()
le_education = LabelEncoder()

bank_data['job'] = le_job.fit_transform(bank_data['job'])
bank_data['marital'] = le_marital.fit_transform(bank_data['marital'])
bank_data['education'] = le_education.fit_transform(bank_data['education'])

# Seleccionar las variables y escalar
X = bank_data[['age', 'job', 'marital', 'education']]
y = bank_data['y']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo 1: Red neuronal original
modelo_1 = Sequential([
    Dense(16, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
modelo_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo_1.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Modelo 2: Red neuronal ajustada
modelo_2 = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
modelo_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo_2.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# 1. Calcular el número total de contactos por mes
total_contacts_per_month = df.groupby('month').size().reset_index(name='Total Contacts')

# 2. Calcular el porcentaje de éxito por mes
success_contacts = df[df['y'] == 1].groupby('month').size().reset_index(name='Success Contacts')
success_rate_per_month = pd.merge(total_contacts_per_month, success_contacts, on='month', how='left')
success_rate_per_month['Success Rate (%)'] = (success_rate_per_month['Success Contacts'] / 
                                              success_rate_per_month['Total Contacts']) * 100

# 3. Calcular el saldo promedio por mes
average_balance_per_month = df.groupby('month')['balance'].mean().reset_index().rename(columns={'balance': 'Average Balance'})

# 4. Calcular la duración promedio de la llamada por mes
average_duration_per_month = df.groupby('month')['duration'].mean().reset_index().rename(columns={'duration': 'Average Duration'})

# Combinar todos los indicadores en un solo DataFrame
indicators_df = pd.merge(total_contacts_per_month, success_rate_per_month[['month', 'Success Rate (%)']], on='month', how='left')\
                  .merge(average_balance_per_month, on='month', how='left')\
                  .merge(average_duration_per_month, on='month', how='left')

# Filtrar solo las columnas sociodemográficas
sociodemographic_cols = ['age', 'job', 'marital', 'education', 'housing', 'loan']

# Crear la app Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout del dashboard
app.layout = dbc.Container([# Imagen en el primer div
    html.Div([
        html.Img(
            src="assets/450_1000.jpg", 
            style={'width': '100%', 'height': 'auto'}
        ),
        html.H2("Visualización de Ventas", className='text-center my-4')
    ], className='mb-4'),
    html.H1("Dashboard de Indicadores de Productos Bancarios", className='text-center my-4'),

    # Filtro de mes en el panel de KPIs
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='month-filter',
                options=[{'label': month.capitalize(), 'value': month} for month in indicators_df['month'].unique()],
                placeholder="Selecciona un mes",
                className='mb-3'
            )
        ], width=6)
    ]),

    # Sección de KPIs
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4('Total de Contactos', className='card-title'),
                html.H2(id='total-contacts', className='card-text')
            ])
        ], color='primary', inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4('Porcentaje de Aceptación', className='card-title'),
                html.H2(id='success-rate', className='card-text')
            ])
        ], color='success', inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4('Saldo Promedio', className='card-title'),
                html.H2(id='average-balance', className='card-text')
            ])
        ], color='info', inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4('Duración Promedio', className='card-title'),
                html.H2(id='average-duration', className='card-text')
            ])
        ], color='warning', inverse=True), width=3)
    ], className='mb-4'),

    html.H2("Perfilamiento Sociodemográfico", className='text-center my-4'),

    # Distribuir el mapa de calor y la gráfica de barras en dos columnas
    dbc.Row([
        dbc.Col([
            html.Label("Selecciona la primera variable:"),
            dcc.Dropdown(
                id='heatmap-var1',
                options=[{'label': col.capitalize(), 'value': col} for col in sociodemographic_cols],
                placeholder="Selecciona la primera variable",
                className='mb-2'
            ),
            html.Label("Selecciona la segunda variable:"),
            dcc.Dropdown(
                id='heatmap-var2',
                options=[{'label': col.capitalize(), 'value': col} for col in sociodemographic_cols],
                placeholder="Selecciona la segunda variable",
                className='mb-2'
            ),
            dcc.Graph(id='heatmap-2-vars', style={'height': '400px', 'width': '100%'})
        ], width=6, style={'padding': '0px'}),

        dbc.Col([
            html.Label("Selecciona una variable para analizar el efecto sobre 'y':"),
            dcc.Dropdown(
                id='bar-chart-var',
                options=[{'label': col.capitalize(), 'value': col} for col in sociodemographic_cols],
                placeholder="Selecciona una variable",
                className='mb-2'
            ),
            dcc.Graph(id='bar-chart-y', style={'height': '400px', 'width': '100%'})
        ], width=6, style={'padding': '0px'})
    ], className='mb-4'),

    html.H2("Análisis de Campaña", className='text-center my-4'),

    # Filtro de mes específico para análisis de campaña
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='campaign-month-filter',
                options=[{'label': month.capitalize(), 'value': month} for month in df['month'].unique()],
                placeholder="Selecciona un mes para el análisis de campaña",
                className='mb-3'
            )
        ], width=6)
    ]),

    # RadioItems para seleccionar el gráfico de análisis de campaña
    dbc.Row([
        dbc.Col([
            dcc.RadioItems(
                id='campaign-analysis-selector',
                options=[
                    {'label': 'Tasa de Conversión', 'value': 'conversion-rate'},
                    {'label': 'Duración de Llamadas', 'value': 'calls-duration'},
                    {'label': 'Llamadas por Día', 'value': 'daily-calls'},
                    {'label': 'Campañas por Cliente', 'value': 'campaigns-per-customer'}
                ],
                value='conversion-rate',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
            dcc.Graph(id='campaign-analysis-graph')
        ], width=12)
    ], className='mb-4'),

    # Nueva sección: Predicción de Ventas
    html.H2("Predicción de Ventas", className='text-center my-4'),

        # Predicción de ventas para Modelo 1 y Modelo 2
    dbc.Row([
        # Predicción Modelo 1
        dbc.Col([
            html.H2("Predicción y Evaluación - Modelo 1", className='text-center mb-2'),
            html.Label('Edad:'),
            dcc.Input(id='input_edad_1', type='number', value=30, style={'backgroundColor': '#f5f5f5'}),
            html.Label('Ocupación:'),
            dcc.Dropdown(
                id='input_ocupacion_1',
                options=[{'label': x, 'value': le_job.transform([x])[0]} for x in le_job.classes_],
                style={'backgroundColor': '#f5f5f5'}
            ),
            html.Label('Estado Civil:'),
            dcc.Dropdown(
                id='input_civil_1',
                options=[{'label': x, 'value': le_marital.transform([x])[0]} for x in le_marital.classes_],
                style={'backgroundColor': '#f5f5f5'}
            ),
            html.Label('Educación:'),
            dcc.Dropdown(
                id='input_educacion_1',
                options=[{'label': x, 'value': le_education.transform([x])[0]} for x in le_education.classes_],
                style={'backgroundColor': '#f5f5f5'}
            ),
            html.Button('Predecir con Modelo 1', id='boton_predecir_1', n_clicks=0,
                        style={'backgroundColor': '#000000', 'color': '#ffffff'}),
            html.Div(id='resultado_prediccion_1', style={'padding': '10px'})
        ], width=6, style={'padding': '10px', 'border': '1px solid #000000'}),

        # Predicción Modelo 2
        dbc.Col([
            html.H2("Predicción y Evaluación - Modelo 2", className='text-center mb-2'),
            html.Label('Edad:'),
            dcc.Input(id='input_edad_2', type='number', value=30, style={'backgroundColor': '#f5f5f5'}),
            html.Label('Ocupación:'),
            dcc.Dropdown(
                id='input_ocupacion_2',
                options=[{'label': x, 'value': le_job.transform([x])[0]} for x in le_job.classes_],
                style={'backgroundColor': '#f5f5f5'}
            ),
            html.Label('Estado Civil:'),
            dcc.Dropdown(
                id='input_civil_2',
                options=[{'label': x, 'value': le_marital.transform([x])[0]} for x in le_marital.classes_],
                style={'backgroundColor': '#f5f5f5'}
            ),
            html.Label('Educación:'),
            dcc.Dropdown(
                id='input_educacion_2',
                options=[{'label': x, 'value': le_education.transform([x])[0]} for x in le_education.classes_],
                style={'backgroundColor': '#f5f5f5'}
            ),
            html.Button('Predecir con Modelo 2', id='boton_predecir_2', n_clicks=0,
                        style={'backgroundColor': '#000000', 'color': '#ffffff'}),
            html.Div(id='resultado_prediccion_2', style={'padding': '10px'})
        ], width=6, style={'padding': '10px', 'border': '1px solid #000000'})
    ])
], fluid=True)

# Callback para actualizar los KPIs
@app.callback(
    [Output('total-contacts', 'children'),
     Output('success-rate', 'children'),
     Output('average-balance', 'children'),
     Output('average-duration', 'children')],
    [Input('month-filter', 'value')]
)
def update_kpis(selected_month):
    filtered_df = indicators_df if selected_month is None else indicators_df[indicators_df['month'] == selected_month]

    total_contacts = filtered_df['Total Contacts'].sum()
    success_rate = filtered_df['Success Rate (%)'].mean()
    average_balance = filtered_df['Average Balance'].mean()
    average_duration = filtered_df['Average Duration'].mean()

    total_contacts_text = f"{total_contacts:,.0f}"
    success_rate_text = f"{success_rate:.2f}%"
    average_balance_text = f"${average_balance:,.2f}"
    average_duration_text = f"{average_duration:.2f} seg"

    return total_contacts_text, success_rate_text, average_balance_text, average_duration_text

# Callback para el mapa de calor de 2 variables
@app.callback(
    Output('heatmap-2-vars', 'figure'),
    [Input('heatmap-var1', 'value'),
     Input('heatmap-var2', 'value')]
)
def update_heatmap(var1, var2):
    if var1 and var2 and var1 != var2:
        heatmap_data = pd.crosstab(df[var1], df[var2])
        fig = px.imshow(
            heatmap_data,
            labels=dict(x=var2.capitalize(), y=var1.capitalize(), color="Frecuencia"),
            title=f"Mapa de Calor: {var1.capitalize()} vs {var2.capitalize()}",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=400)
    else:
        fig = go.Figure()
    return fig

# Callback para la gráfica de barras de efecto sobre 'y'
@app.callback(
    Output('bar-chart-y', 'figure'),
    [Input('bar-chart-var', 'value'),
     Input('month-filter', 'value')]
)
def update_bar_chart(var, selected_month):
    if var:
        df_filtered = df if selected_month is None else df[df['month'] == selected_month]
        bar_data = pd.crosstab(df_filtered[var], df_filtered['y'])
        bar_data.columns = ['No Exitoso', 'Exitoso']
        bar_data = bar_data.reset_index()
        fig = px.bar(
            bar_data,
            x=var,
            y=['No Exitoso', 'Exitoso'],
            title=f"Efecto de {var.capitalize()} sobre 'y'",
            labels={'value': 'Frecuencia', var: var.capitalize()},
            barmode='group'
        )
        fig.update_layout(height=400)
    else:
        fig = go.Figure()
    return fig

# Callback para el análisis de campaña con filtro de mes
@app.callback(
    Output('campaign-analysis-graph', 'figure'),
    [Input('campaign-analysis-selector', 'value'),
     Input('campaign-month-filter', 'value')]
)
def update_campaign_graph(selected_analysis, selected_month):
    df_filtered = df if selected_month is None else df[df['month'] == selected_month]

    if selected_analysis == 'conversion-rate':
        conversion_rate = (df_filtered['y'].sum() / df_filtered.shape[0]) * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conversion_rate,
            title={"text": "Tasa de Conversión (%)"}
        ))

    elif selected_analysis == 'calls-duration':
        duration_per_day = df_filtered.groupby('day')['duration'].mean().reset_index()
        fig = px.line(duration_per_day, x='day', y='duration',
                      title='Duración Media de Llamadas por Día')

    elif selected_analysis == 'daily-calls':
        daily_calls = df_filtered.groupby('day').size().reset_index(name='calls')
        fig = px.line(daily_calls, x='day', y='calls', title='Número de Llamadas por Día')

    elif selected_analysis == 'campaigns-per-customer':
        campaigns_per_customer = df_filtered.groupby('contact').size().reset_index(name='campaigns')
        fig = px.bar(campaigns_per_customer, x='contact', y='campaigns',
                     title='Número de Campañas por Cliente')

    else:
        fig = go.Figure()

    return fig

# Callback para predicción con Modelo 1
@app.callback(
    Output('resultado_prediccion_1', 'children'),
    Input('boton_predecir_1', 'n_clicks'),
    State('input_edad_1', 'value'),
    State('input_ocupacion_1', 'value'),
    State('input_civil_1', 'value'),
    State('input_educacion_1', 'value')
)
def predecir_conversion_1(n_clicks, edad, ocupacion, civil, educacion):
    if n_clicks > 0:
        X_nuevo = pd.DataFrame([[edad, ocupacion, civil, educacion]], columns=['age', 'job', 'marital', 'education'])
        X_nuevo_scaled = scaler.transform(X_nuevo)
        probabilidad = modelo_1.predict(X_nuevo_scaled)[0][0]
        resultado = f"Modelo 1: {'Alta' if probabilidad > 0.5 else 'Baja'} probabilidad de conversión ({probabilidad:.2f})"
        return resultado

# Callback para predicción con Modelo 2
@app.callback(
    Output('resultado_prediccion_2', 'children'),
    Input('boton_predecir_2', 'n_clicks'),
    State('input_edad_2', 'value'),
    State('input_ocupacion_2', 'value'),
    State('input_civil_2', 'value'),
    State('input_educacion_2', 'value')
)
def predecir_conversion_2(n_clicks, edad, ocupacion, civil, educacion):
    if n_clicks > 0:
        X_nuevo = pd.DataFrame([[edad, ocupacion, civil, educacion]], columns=['age', 'job', 'marital', 'education'])
        X_nuevo_scaled = scaler.transform(X_nuevo)
        probabilidad = modelo_2.predict(X_nuevo_scaled)[0][0]
        resultado = f"Modelo 2: {'Alta' if probabilidad > 0.5 else 'Baja'} probabilidad de conversión ({probabilidad:.2f})"
        return resultado


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)











