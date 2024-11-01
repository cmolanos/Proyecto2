import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Cargar la base de datos
file_path = 'Bank.csv'
df = pd.read_csv(file_path, delimiter=';')
df.columns = df.columns.str.strip().str.lower()

# Manejo de valores nulos y codificación de variables
df.fillna('unknown', inplace=True)

# Mapear 'month' a números y crear 'month_no' para el modelo
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
df['month_no'] = df['month'].map(month_mapping)

# Codificación de variables categóricas para modelos 1 y 2
le_job = LabelEncoder()
le_marital = LabelEncoder()
le_education = LabelEncoder()
le_contact = LabelEncoder()
le_poutcome = LabelEncoder()

# Aplicar transformaciones y guardar mapeos
df['job'] = le_job.fit_transform(df['job'])
df['marital'] = le_marital.fit_transform(df['marital'])
df['education'] = le_education.fit_transform(df['education'])
df['contact'] = le_contact.fit_transform(df['contact'])
df['poutcome'] = le_poutcome.fit_transform(df['poutcome'])

category_mappings = {
    'job': dict(enumerate(le_job.classes_)),
    'marital': dict(enumerate(le_marital.classes_)),
    'education': dict(enumerate(le_education.classes_)),
    'contact': dict(enumerate(le_contact.classes_)),
    'poutcome': dict(enumerate(le_poutcome.classes_))
}


# Variables para el Modelo 1
X1 = df[['age', 'job', 'marital', 'education', 'balance', 'loan', 'housing']]
y1 = df['y']

# Variables para el Modelo 2 (usando 'month_no')
X2 = df[['campaign', 'day', 'contact', 'poutcome', 'month_no', 'duration', 'pdays']]
y2 = df['y']

# Escalar las características
scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

# Dividir los datos para Modelo 1
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, test_size=0.2, random_state=42)

# Dividir los datos para Modelo 2
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42)

# Modelo 1: Red neuronal
modelo_1 = Sequential([
    Dense(16, input_dim=X1_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
modelo_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo_1.fit(X1_train, y1_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Modelo 2: Red neuronal
modelo_2 = Sequential([
    Dense(32, input_dim=X2_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
modelo_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo_2.fit(X2_train, y2_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
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


# Crear la app Dash con un tema claro
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout del dashboard
app.layout = dbc.Container([
    # Imagen en el encabezado
    html.Div([
           html.Img(
        src="assets/download.png",
        style={
            'width': '100%', 
            'height': '350px',  # Ajuste de altura a 200px
            'object-fit': 'cover',
            'border-radius': '10px'
        }
    )
    ], className='mb-4 p-4 rounded'),

    # Título principal del dashboard
    html.H1("Dashboard de Indicadores de Productos Bancarios",
            className='text-center my-4 text-dark p-3 rounded'),

    # Filtro de mes en el panel de KPIs
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='month-filter',
                options=[{'label': month.capitalize(), 'value': month} for month in df['month'].unique()],
                placeholder="Selecciona un mes",
                className='mb-3',
                style={'color': '#000000'}
            )
        ], width=6)
    ], className='mb-4'),

    # Sección de KPIs
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4('Total de Contactos', className='card-title text-dark'),
                html.H2(id='total-contacts', className='card-text text-dark')
            ])
        ], color='light', inverse=False, style={'border-radius': '10px'}), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4('Porcentaje de Aceptación', className='card-title text-dark'),
                html.H2(id='success-rate', className='card-text text-dark')
            ])
        ], color='light', inverse=False, style={'border-radius': '10px'}), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4('Saldo Promedio', className='card-title text-dark'),
                html.H2(id='average-balance', className='card-text text-dark')
            ])
        ], color='light', inverse=False, style={'border-radius': '10px'}), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4('Duración Promedio', className='card-title text-dark'),
                html.H2(id='average-duration', className='card-text text-dark')
            ])
        ], color='light', inverse=False, style={'border-radius': '10px'}), width=3)
    ], className='mb-4'),

    # Título de perfilamiento sociodemográfico
    html.H2("Perfilamiento Sociodemográfico", className='text-center my-4 text-dark p-3 rounded'),

    # Mapa de calor y gráfica de barras
    dbc.Row([
        dbc.Col([
            html.Label("Selecciona la primera variable:", className='text-dark mb-2'),
            dcc.Dropdown(
                id='heatmap-var1',
                options=[{'label': col.capitalize(), 'value': col} for col in sociodemographic_cols],
                placeholder="Selecciona la primera variable",
                className='mb-3',
                style={'color': '#000000'}
            ),
            html.Label("Selecciona la segunda variable:", className='text-dark mb-2'),
            dcc.Dropdown(
                id='heatmap-var2',
                options=[{'label': col.capitalize(), 'value': col} for col in sociodemographic_cols],
                placeholder="Selecciona la segunda variable",
                className='mb-3',
                style={'color': '#000000'}
            ),
            dcc.Graph(id='heatmap-2-vars', style={'height': '400px', 'width': '100%'})
        ], width=6, style={'padding': '10px'}),

        dbc.Col([
            html.Label("Selecciona una variable para analizar el efecto sobre 'y':", className='text-dark mb-2'),
            dcc.Dropdown(
                id='bar-chart-var',
                options=[{'label': col.capitalize(), 'value': col} for col in sociodemographic_cols],
                placeholder="Selecciona una variable",
                className='mb-3',
                style={'color': '#000000'}
            ),
            dcc.Graph(id='bar-chart-y', style={'height': '400px', 'width': '100%'})
        ], width=6, style={'padding': '10px'})
    ], className='mb-4'),

    # Análisis de Campaña
    html.H2("Análisis de Campaña", className='text-center my-4 text-dark p-3 rounded'),

    # Filtro de mes específico para análisis de campaña
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='campaign-month-filter',
                options=[{'label': month.capitalize(), 'value': month} for month in df['month'].unique()],
                placeholder="Selecciona un mes para el análisis de campaña",
                className='mb-3',
                style={'color': '#000000'}
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
                labelStyle={'display': 'inline-block', 'margin-right': '10px', 'color': '#000000'}
            ),
            dcc.Graph(id='campaign-analysis-graph', style={'height': '400px'})
        ], width=12)
    ], className='mb-4'),

    # Predicción de Ventas
    html.H2("Predicción de Ventas", className='text-center my-4 text-dark p-3 rounded'),

    # Formulario de predicción para Modelo 1 y Modelo 2
    dbc.Row([
        # Formulario de Modelo 1
        dbc.Col([
            html.H2("Predicción y Evaluación - Modelo 1", className='text-center mb-4 text-dark'),
            
            html.Label('Edad:', className='text-dark mt-2'),
            dcc.Input(id='input_edad_1', type='number', value=30, className='form-control mb-3'),

            html.Label('Ocupación:', className='text-dark mt-2'),
            dcc.Dropdown(
                id='input_ocupacion_1',
                options=[{'label': x, 'value': le_job.transform([x])[0]} for x in le_job.classes_],
                className='mb-3',
                style={'color': '#000000'}
            ),
              

            html.Label('Estado Civil:', className='text-dark mt-2'),
            dcc.Dropdown(
                id='input_civil_1',
                options=[{'label': x, 'value': le_marital.transform([x])[0]} for x in le_marital.classes_],
                className='mb-3',
                style={'color': '#000000'}
            ),

            html.Label('Educación:', className='text-dark mt-2'),
            dcc.Dropdown(
                id='input_educacion_1',
                options=[{'label': x, 'value': le_education.transform([x])[0]} for x in le_education.classes_],
                className='mb-3',
                style={'color': '#000000'}
            ),

            html.Label('Saldo:', className='text-dark mt-2'),
            dcc.Input(id='input_balance_1', type='number', value=0, className='form-control mb-3'),

            html.Label('Préstamo Personal:', className='text-dark mt-2'),
            dcc.Dropdown(
                id='input_loan_1',
                options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
                className='mb-3',
                style={'color': '#000000'}
            ),

            html.Label('Préstamo Hipotecario:', className='text-dark mt-2'),
            dcc.Dropdown(
                id='input_housing_1',
                options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
                className='mb-3',
                style={'color': '#000000'}
            ),

            html.Button('Predecir con Modelo 1', id='boton_predecir_1', n_clicks=0,
                        className='btn btn-primary mt-3 mb-3'),
            html.Div(id='resultado_prediccion_1', className='mt-3 text-dark')
        ], width=6, className='p-4'),

        # Formulario de Modelo 2
        dbc.Col([
            html.H2("Predicción y Evaluación - Modelo 2", className='text-center mb-4 text-dark'),
            
            html.Label('Número de Campañas:', className='text-dark mt-2'),
            dcc.Input(id='input_campaign_2', type='number', value=1, className='form-control mb-3'),

            html.Label('Día de Contacto:', className='text-dark mt-2'),
            dcc.Input(id='input_day_2', type='number', value=1, className='form-control mb-3'),

            html.Label('Tipo de Contacto:', className='text-dark mt-2'),
            dcc.Dropdown(
                id='input_contact_2',
                options=[{'label': x, 'value': le_contact.transform([x])[0]} for x in le_contact.classes_],
                className='mb-3',
                style={'color': '#000000'}
            ),

            html.Label('Resultado de Campaña Anterior:', className='text-dark mt-2'),
            dcc.Dropdown(
                id='input_poutcome_2',
                options=[{'label': x, 'value': le_poutcome.transform([x])[0]} for x in le_poutcome.classes_],
                className='mb-3',
                style={'color': '#000000'}
            ),

            html.Label('Mes del Contacto:', className='text-dark mt-2'),
            dcc.Dropdown(
                id='input_month_2',
                options=[{'label': x.capitalize(), 'value': month_mapping[x]} for x in df['month'].unique()],
                className='mb-3',
                style={'color': '#000000'}
            ),

            html.Label('Duración del Contacto:', className='text-dark mt-2'),
            dcc.Input(id='input_duration_2', type='number', value=0, className='mb-3'),

            html.Label('Días desde el último contacto:', className='text-dark mt-2'),
            dcc.Input(id='input_pdays_2', type='number', value=-1, className='mb-3'),

            html.Button('Predecir con Modelo 2', id='boton_predecir_2', n_clicks=0,
                        className='btn btn-primary mt-3 mb-3'),
            html.Div(id='resultado_prediccion_2', className='mt-3 text-dark')
        ], width=6, className='p-4')
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
        # Crear una copia solo si es necesario
        heatmap_df = df.copy()
        
        # Mapear etiquetas para variables categóricas
        if var1 in category_mappings:
            heatmap_df[var1] = heatmap_df[var1].map(category_mappings[var1])
        if var2 in category_mappings:
            heatmap_df[var2] = heatmap_df[var2].map(category_mappings[var2])
        
        # Crear la tabla de frecuencia cruzada
        heatmap_data = pd.crosstab(heatmap_df[var1], heatmap_df[var2])
        
        # Crear el heatmap con Plotly
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
        # Filtrar el DataFrame según el mes seleccionado
        df_filtered = df if selected_month is None else df[df['month'] == selected_month]
        
        # Crear una copia solo si es necesario para mapear etiquetas categóricas
        bar_chart_df = df_filtered.copy()
        
        # Mapear etiquetas si la variable es categórica
        if var in category_mappings:
            bar_chart_df[var] = bar_chart_df[var].map(category_mappings[var])
        
        # Crear la tabla de frecuencia cruzada
        bar_data = pd.crosstab(bar_chart_df[var], bar_chart_df['y'])
        bar_data.columns = ['No Exitoso', 'Exitoso']
        bar_data = bar_data.reset_index()
        
        # Crear la gráfica de barras con Plotly
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

 #Callback para predicción con Modelo 1
@app.callback(
    Output('resultado_prediccion_1', 'children'),
    Input('boton_predecir_1', 'n_clicks'),
    State('input_edad_1', 'value'),
    State('input_ocupacion_1', 'value'),
    State('input_civil_1', 'value'),
    State('input_educacion_1', 'value'),
    State('input_balance_1', 'value'),
    State('input_loan_1', 'value'),
    State('input_housing_1', 'value')
)
def predecir_conversion_1(n_clicks, edad, ocupacion, civil, educacion, balance, loan, housing):
    if n_clicks > 0:
        X_nuevo = pd.DataFrame([[edad, ocupacion, civil, educacion, balance, loan, housing]], 
                               columns=['age', 'job', 'marital', 'education', 'balance', 'loan', 'housing'])
        X_nuevo_scaled = scaler1.transform(X_nuevo)
        probabilidad = modelo_1.predict(X_nuevo_scaled)[0][0]
        resultado = f"Modelo 1: {'Alta' if probabilidad > 0.5 else 'Baja'} probabilidad de conversión ({probabilidad:.2f})"
        return resultado

# Callback para predicción con Modelo 2
@app.callback(
    Output('resultado_prediccion_2', 'children'),
    Input('boton_predecir_2', 'n_clicks'),
    State('input_campaign_2', 'value'),
    State('input_day_2', 'value'),
    State('input_contact_2', 'value'),
    State('input_poutcome_2', 'value'),
    State('input_month_2', 'value'),
    State('input_duration_2', 'value'),
    State('input_pdays_2', 'value')
)
def predecir_conversion_2(n_clicks, campaign, day, contact, poutcome, month, duration, pdays):
    if n_clicks > 0:
        # No es necesario mapear 'month', ya está en numérico
        month_no = month

        # Validar que 'month_no' sea un número válido
        if month_no is None or not isinstance(month_no, int):
            return "Error: El mes seleccionado no es válido."

        X_nuevo = pd.DataFrame([[campaign, day, contact, poutcome, month_no, duration, pdays]], 
                               columns=['campaign', 'day', 'contact', 'poutcome', 'month_no', 'duration', 'pdays'])
        
        X_nuevo_scaled = scaler2.transform(X_nuevo)
        probabilidad = modelo_2.predict(X_nuevo_scaled)[0][0]
        resultado = f"Modelo 2: {'Alta' if probabilidad > 0.5 else 'Baja'} probabilidad de conversión ({probabilidad:.2f})"
        return resultado

    return ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)