from flask import Flask, render_template, make_response, jsonify, request
import pandas
import plotly.graph_objs as go
import plotly.express as px
from joblib import dump, load
app = Flask(__name__)

dff = pandas.read_csv('analysis.csv')
df = pandas.read_csv('dataset.csv')
cyber_data = pandas.read_csv('cyber.csv')

@app.route('/api', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        name = request.get_json()['name']
        row = df[df['State/UT'].str.contains(name, case=False)]
        full_data = row.to_dict(orient='records')[0]
        graph_row = row.drop(['Total Crimes', 'State/UT'], axis=1)
        graph_data = graph_row.to_dict(orient='records')[0]

        graph_arr = [['Year', 'Cases']]
        for key, val in graph_data.items():
            graph_arr.append([key, val])

        return make_response(jsonify({
            'name': full_data['State/UT'],
            'data': full_data,
            'graph': graph_arr
        }))
    else:
        return make_response(jsonify({'error': 'UnAuthorised!'}))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/CA')
def CA():
    return render_template('CA.html')

@app.route('/graph', methods=['POST'])
def graph():
    # get the selected year and type of cybercrime from the form
    year = int(request.form['year'])
    cybercrime = request.form['cybercrime']
    state = request.form['state']
    plot_type = request.form['plot-type']

    error = []
    if plot_type == 'crime-types':
        if not year or not cybercrime:
            error.append('Please select year and cybercrime type.')
            return render_template('graph.html', error=error[0])
        else:
            df_year = dff[dff['Year'] == year]
            cyber_crime = df_year.groupby('State')[cybercrime].sum().reset_index()
            fig = px.bar(cyber_crime, x='State', y=cybercrime, title=f"{cybercrime} Cases in {year}")
            div = fig.to_html(full_html=False)
            return render_template('graph.html', plot_div=div)
    elif plot_type == 'state-count':
        if not year or not state:
            error.append('Please select year and state.')
            return render_template('graph.html', error=error[0])
        else:
            data_2017_MH = dff[(dff['Year'] == year) & (dff['State'] == state)]
            df_grouped = data_2017_MH.groupby('State')[
                ['Identity theft', 'Forgery (Sec.465,468 & 471)', 'Cyber Stalking', 'ATM',
                 'personation by using computer']].sum().reset_index()
            df_melted = df_grouped.melt(id_vars=['State'],
                                        value_vars=['Identity theft', 'Forgery (Sec.465,468 & 471)', 'Cyber Stalking',
                                                    'ATM', 'personation by using computer'], var_name='Crime Type',
                                        value_name='Count')
            fig = px.bar(df_melted, x='State', y='Count', color='Crime Type', barmode='group',
                         title=f'cybercrimes and count in {year} for {state}')
            div = fig.to_html(full_html=False)
            return render_template('graph.html', plot_div=div)

    elif plot_type == 'line-graph':
        df = pandas.read_csv('analysis.csv')
        crime_heads = ['Identity theft', 'Forgery (Sec.465,468 & 471)', 'Cyber Stalking', 'ATM',
                       'personation by using computer']
        df = df[['Year'] + crime_heads]
        df = df.groupby('Year').sum().reset_index()
        traces = []
        for crime_head in crime_heads:
            trace = go.Scatter(x=df['Year'], y=df[crime_head], mode='lines', name=crime_head)
            traces.append(trace)
        layout = go.Layout(title='Trend of Cybercrimes over the Years', xaxis_title='Year',
                           yaxis_title='Number of Crimes')
        fig = go.Figure(data=traces, layout=layout)
        div = fig.to_html(full_html=False)
        return render_template('graph.html', plot_div=div)

    elif plot_type == 'pie-graph':
        if not year or not state:
            error.append('Please select state.')
            return render_template('graph.html', error=error[0])
        else:
            data = pandas.read_csv('analysis.csv')
            state = request.form['state']
            maharashtra_data = data[data['State'] == state]
            fig = go.Figure(data=[go.Pie(labels=['Total Internet Subscriptions', 'Total Broadband Subscriptions',
                                                 'Total Wireless internet Subscriptions'],
                                         values=[maharashtra_data['Total Internet Subscriptions'].values[0],
                                                 maharashtra_data['Total Broadband Subscriptions'].values[0],
                                                 maharashtra_data['Total Wireless internet Subscriptions'].values[0]])])

            # set chart title
            fig.update_layout(title=f'Distribution of Internet Subscriptions in {state}')
            div = fig.to_html(full_html=False)
            return render_template('graph.html', plot_div=div)

    else:
        pass

def find_max_key(value: int, all_years_predictions: dict) -> str:
    for key, val in all_years_predictions.items():
        if val == value:
            return key

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        state_info = request.get_json()['state_info']
        state = state_info['state']
        state_info.pop('state')
        sample = [int(val) for key, val in state_info.items()]
        all_states = list(set(cyber_data['State'].to_list()))
        all_states.sort()
        state_df = cyber_data[cyber_data['State'] == state]
        years_known = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
        years_data_known = state_df['Cases Reported'].to_list()

        years_display = [x for x in years_known]
        years_display.append(int(state_info['year']))

        state_data = state_df.to_dict(orient='list')
        rm_keys = ['Unique Code', 'State']
        for key in rm_keys:
            state_data.pop(key)
        path = f'models/{state}.joblib'
        loaded_model = load(path)
        # sample_input = [[2022, 90000000, 60000000, 57000000, 59000000]]
        result = loaded_model.predict([sample])

        X = state_df.drop(['Unique Code', 'State', 'Cases Reported'], axis=1)
        years_data_display = list(loaded_model.predict(X.values))
        years_data_display.append(result[0])

        years_data_display = [int(x) for x in years_data_display]

        all_years_predictions = {}
        all_predictions = []
        for i in all_states:
            path = f'models/{i}.joblib'
            model = load(path)
            value = int(model.predict([sample])[0])
            all_years_predictions[i] = value
            all_predictions.append(value)

        max_val = max(all_predictions)

        return make_response(jsonify(
            {
                'state_data': state_data,
                'result': result[0],
                'years_known': years_known,
                'years_data_known': years_data_known,
                'years_display': years_display,
                'years_data_display': years_data_display,
                'all_years_predictions': all_years_predictions,
                'max_predictions': find_max_key(max_val, all_years_predictions)
            }
        ))
    else:
        return render_template('predictions.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
