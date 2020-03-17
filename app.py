import os
from flask_bootstrap import Bootstrap
from flask import Flask, render_template, request, flash, redirect, url_for
from hypothesisTest.performance import metrics
import hypothesisTest.settings as st
import hypothesisEngine.hypothesis_engine as hy_engine
import hypothesisEngine.print_setting as ps
import numpy as np
from hypothesisEngine.stats.stats import get_stats
from hypothesisEngine.algorithm.parameters import params, params1, set_params
import sys
#sys.stdout=open("test.txt","a")
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

#////////////////////////////////////////////////////////////////////////////////
app = Flask(__name__)
Bootstrap(app)


#////////////////////////////////////////////////////////////////////////////////
class cls_settings:
    def __init__(self):
        self.start_date = []
        self.settings_values.end_date = []
        self.settings_values.portfolio = []
        self.settings_values.dollar_neutral = []
        self.settings_values.long_leverage = []
        self.settings_values.short_leverage = []
        self.settings_values.costs_threshold = []
        self.settings_values.starting_value =  []
        self.settings_values.strategy_expression = []


class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        pass

#////////////////////////////////////////////////////////////////////////////////
global settings_values

settings_values = cls_settings
settings_values.start_date = "2004-01-01"
settings_values.end_date = "2018-11-15"
settings_values.portfolio = 0
settings_values.dollar_neutral = 0
settings_values.long_leverage = "0.5"
settings_values.short_leverage = "0.5"
settings_values.costs_threshold = "0"
settings_values.starting_value = "20000000"
settings_values.strategy_expression = '-rank(Volume)*(gauss_filter(High,5)-gauss_filter(Open,5))'

ps.print_list = []

#////////////////////////////////////////////////////////////////////////////////
def start(filename):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Transcript(filename)

#////////////////////////////////////////////////////////////////////////////////
def stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal

#////////////////////////////////////////////////////////////////////////////////
@app.route("/")
def home():
    return render_template('home.html')

#////////////////////////////////////////////////////////////////////////////////
@app.route("/test_hypothesis", methods=['GET', 'POST'])
def test_hypothesis():
    res = {}
    # res['curves'] = {}
    # res['curves']['Strategy'] = [0]
    # res['curves']['Benchmark'] = [0]
    plot_data = [[[0,0]],[[0,0]]]

    if request.method == 'POST':
        st.start_date = request.form['start_date']
        st.end_date = request.form['end_date']

        if  request.form['dollar_neutral']=="0":
            st.dollar_neutral = True
        else:
            st.dollar_neutral = False

        st.long_leverage = np.float(request.form['long_leverage'])
        st.short_leverage = np.float(request.form['short_leverage'])
        st.costs_threshold = np.float(str(request.form['costs_threshold']).replace('%',''))
        st.starting_value = np.float(str(request.form['starting_value']).replace('$',''))
        st.strategy_expression = request.form['strategy_expression']

        settings_values.start_date = st.start_date
        settings_values.end_date = st.end_date
        settings_values.portfolio = np.int16(request.form['portfolio'])
        settings_values.dollar_neutral = np.int16(request.form['dollar_neutral'])
        settings_values.long_leverage = request.form['long_leverage']
        settings_values.short_leverage = request.form['short_leverage']
        settings_values.costs_threshold = request.form['costs_threshold']
        settings_values.starting_value =  request.form['starting_value']
        settings_values.strategy_expression = st.strategy_expression

        res, plot_data = metrics()
        for key, value in res.items():
            if key not in ["CLASSIFICATION_DATA", "FACTOR_RES", "cleaned_index", "curves" ]:
                print("{:<40}{:^5}{:<20}".format(key, " :\t", value))

        print("Classification Metrics : \n")
        print(res['CLASSIFICATION_DATA'], "\n")
        print("Factor Analysis : \n")
        print(res['FACTOR_RES'])

        return render_template('test_hypothesis.html', settings_values=settings_values, res=res, plot_data=plot_data, confusionFlag = True)
    #
    # plot_data = [list(res['curves']['Strategy']), list(res['curves']['Benchmark'])]
    return render_template('test_hypothesis.html', settings_values=settings_values, res=res, plot_data=plot_data, confusionFlag = False )


# ///////////////////////////////////////////////////////////////////////////////////////////
@app.route("/hypothesis_engine", methods=['GET', 'POST'])
def hypothesis_engine():
    if request.method == 'POST':
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        start('log_' + dt_string + '.txt')

        old_keys = list(params1.keys())
        new_keys = list(params.keys())

        for n in range(len(new_keys)):
            if not( old_keys.__contains__(new_keys[n])):
                params.pop(new_keys[n])

        for key in params1.keys():
            params[key] = params1[key]

        params['GRAMMAR_FILE'] = request.values['GRAMMAR_FILE']
        params['FITNESS_FUNCTION'] = request.values['FITNESS_FUNCTION']
        params['MAX_INIT_TREE_DEPTH'] = np.int(request.values['MAX_INIT_TREE_DEPTH'])
        params['MIN_INIT_TREE_DEPTH'] = np.int(request.values['MIN_INIT_TREE_DEPTH'])
        params['INIT_GENOME_LENGTH'] = np.int(request.values['INIT_GENOME_LENGTH'])
        params['INTERACTION_PROBABILITY'] = np.float(request.values['INTERACTION_PROBABILITY'])
        params['MAX_TREE_DEPTH'] = np.int(request.values['MAX_TREE_DEPTH'])
        params['POPULATION_SIZE'] = np.int(request.values['POPULATION_SIZE'])
        params['SELECTION_PROPORTION'] = np.float(request.values['SELECTION_PROPORTION'])
        params['GENERATIONS'] = np.int(request.values['GENERATIONS'])
        params['GENERATION_SIZE'] = np.int(request.values['GENERATION_SIZE'])
        params['ELITE_SIZE'] = np.int(request.values['ELITE_SIZE'])
        params['CROSSOVER_PROBABILITY'] = np.float(request.values['CROSSOVER_PROBABILITY'])

        params['MUTATION_EVENTS'] = np.int(request.values['MUTATION_EVENTS'])
        params['MUTATION_PROBABILITY'] = np.float(request.values['MUTATION_PROBABILITY'])
        params['TOURNAMENT_SIZE'] = np.int(request.values['TOURNAMENT_SIZE'])

        set_params(sys.argv[1:])
        individuals = params['SEARCH_LOOP']()
        get_stats(individuals, end=True)
        # stop()

        return_str = ""
        for n in range(len(ps.print_list)):
            return_str +="<p>"+ps.print_list[n]+"</p>\n"
            # if n==0:
            #     return_str = ps.print_list[n]
            # else:
            #     return_str += ";"+ps.print_list[n]

        return return_str
    else:
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        start('log_' + dt_string + '.txt')

        params['GRAMMAR_FILE'] = "trading_grammar/trading_engine.bnf"
        params['FITNESS_FUNCTION'] = "trading_fitness.trading_engine"
        params['MAX_INIT_TREE_DEPTH'] = 10
        params['MIN_INIT_TREE_DEPTH'] = 3
        params['INIT_GENOME_LENGTH'] = 200
        params['INTERACTION_PROBABILITY'] = 0.5
        params['MAX_TREE_DEPTH'] = 10
        params['POPULATION_SIZE'] = 10
        params['SELECTION_PROPORTION'] = 0.25
        params['GENERATIONS'] = 5
        params['GENERATION_SIZE'] = 3
        params['ELITE_SIZE'] = 1
        params['CROSSOVER_PROBABILITY'] = 0.4

        params['MUTATION_EVENTS'] = 1
        params['MUTATION_PROBABILITY'] = 0.1
        params['TOURNAMENT_SIZE'] = 2

        # set_params(sys.argv[1:])
        #
        # individuals = params['SEARCH_LOOP']()
        #
        # # Print final review
        # get_stats(individuals, end=True)
        #
        # stop()

    return render_template('hypothesisEngine.html', params=params)



# ///////////////////////////////////////////////////////////////////////////////////////////
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

# ///////////////////////////////////////////////////////////////////////////////////////////
@app.route('/simulate', methods=['POST'])
def upload_file():
    global data
    global path_to_file
    global m_labels

    if request.method == 'POST':
        m_str=""
        # f = request.files['file']
        # path_to_file = os.path.join(app.config['UPLOAD_FOLDER'], '1.csv')
        # f.save(os.path.join(app.config['UPLOAD_FOLDER'], '1.csv'))
        # # csvfile = Csv(filename=secure_filename(f.filename))
        # # db.session.add(csv)
        # # db.session.commit()
        # data = pd.read_csv(path_to_file)
        # head_data = np.array(data.head(10))
        # m_size = np.shape(head_data)
        # m_labels = list(data)

        # #------------------------------------------------------
        # m_str = '<div class="limiter"><div><div><div><table>'
        # for m in range(len(m_labels)):
        #     if m == 0: m_str +='<thead><tr class="table100-head">'
        #     m_str += '<th class="columns">' +m_labels[m]+"</th>"
        #     if m == len(m_labels)-1: m_str +='</tr></thead>'

        # # ------------------------------------------------------
        # m_str+='<tbody>'
        # for n in range(m_size[0]):
        #     for m in range(len(m_labels)):
        #         if m == 0: m_str += '<tr>'
        #         m_str += '<td class="columns">' +str( head_data[n][m] )+ "</td>"
        #         if m == len(m_labels) - 1: m_str += '</tr>'

        # m_str +="</tbody></table></div></div></div></div>"

        # # res = {}
    return m_str



def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path, endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)



if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='localhost', port=port)
    app.run(debug=True)