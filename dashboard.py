import dash
from dash import dcc, html, Input, Output
import requests

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Sentiment Analysis Dashboard"),
    dcc.Textarea(id='input-text', placeholder="Enter text...", style={'width': '100%', 'height': 100}),
    html.Button('Analyze', id='analyze-btn', n_clicks=0),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Input('analyze-btn', 'n_clicks'),
    Input('input-text', 'value')
)
def update_output(n_clicks, text):
    if n_clicks > 0 and text:
        response = requests.post("http://127.0.0.1:5000/analyze", json={"text": text})
        result = response.json()

        return html.Div([
            html.P(f"Sentiment: {result['sentiment']} (Score: {result['sentiment_score']:.2f})"),
            html.P(f"Emotion: {result['emotion']} (Score: {result['emotion_score']:.2f})")
        ])
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
