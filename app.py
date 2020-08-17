from flask import Flask, render_template, request
import ml
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
matplotlib.use('Agg')

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():

    h0 = float(request.form.get('hours0'))
    s0 = ml.predicts(h0)
    goal = float(request.form.get('goal'))
    diff = goal-s0
    s0 = round(s0, 2)
    diff = round(diff, 2)
    msg = ""
    if diff > 0:
        msg = "Sorry, not enough to reach your goal..Boost yourself! 💪"
    else:
        msg = "Great going!! Keep it up.."

    data = {'Your goal': goal, 'Predicted': s0}
    d = list(data.keys())
    score = list(data.values())
    # creating the bar plot
    plt.bar(d, score, color='#c8d5b9', ec="black")

    plt.ylabel("Scores")
    plt.title("Goal Vs. Predicted score")
    plt.tight_layout()

    '''
    Old graph fix, filename is now generated from timestamp to avoid collisions.
    '''
    filename = datetime.timestamp(datetime.now())

    # plt.savefig('static/{}.jpg'.format(COUNT))
    # img_name = '{}.jpg'.format(COUNT)
    plt.savefig('static/{}.jpg'.format(filename))
    img_name = '{}.jpg'.format(filename)
    plt.close()

    return render_template("result.html", scores=s0, diff=diff, msg=msg, hour=h0, filename=img_name)

if __name__ == '__main__':
    app.run(debug=True)