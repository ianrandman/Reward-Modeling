from flask import Flask, render_template
from flask import request
from flask import jsonify
import json


def get_pref_db(env):
    with open("preferences/"+env+"/pref_db.json") as f:
        f = open("preferences/" + env + "/pref_db.json")
        return json.load(f)

def save_pref_db(pref_db, env):
    with open("preferences/"+env+'/pref_db.json', 'w') as json_file:
        json.dump(pref_db, json_file)


def get_webapp(trajectory_builder):
    app = Flask(__name__)

    # PAGE ROUTES
    @app.route("/cartpole")
    def env():
        return render_template('env.html', env_name="cartpole")

    @app.route("/")
    def main():
        return render_template('index.html')

    # API ROUTES
    @app.route("/getpair")
    def get_pair():
        data = json.dumps(trajectory_builder.get_pair())
        return data

    @app.route('/preference', methods=['POST'])
    def update_text():
        req = request.json
        env = req["env"]
        del req['env']
        pref_db = get_pref_db(env)
        pref_db['preferences'].append(req)
        save_pref_db(pref_db, env)
        return get_pair()

    return app
