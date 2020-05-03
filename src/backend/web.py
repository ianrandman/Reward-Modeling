from flask import Flask, render_template
from flask import request
import json
import threading


def synchronized(func):
    func.__lock__ = threading.Lock()

    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)
    return synced_func


@synchronized
def checkpoint():
    save_pref_db()


@synchronized
def add_pref(pref, prefs):
    prefs.append(pref)


@synchronized
def save_pref_db(pref_db, env):
    with open("preferences/"+env+'/pref_db.json', 'w') as json_file:
        json.dump(pref_db, json_file)


def get_pref_db(env):
    with open("preferences/"+env+"/pref_db.json") as f:
        f = open("preferences/" + env + "/pref_db.json")
        pref_db = json.load(f)
        return pref_db if len(pref_db) > 0 else None


def get_webapp(trajectory_builder, env_lst):
    app = Flask(__name__)

    db_for_env = {}
    for env in env_lst:
        pref_db = get_pref_db(env)
        if pref_db is None:
            pref_db = []
            db_for_env[env] = pref_db
    #
    #
    #
    #
    #
    # PAGE ROUTES
    @app.route("/cartpole")
    def cartpole():
        return render_template('env.html', env_name="CartPole-v0")

    @app.route("/mountaincar")
    def mountaincar():
        return render_template('env.html', env_name="MountainCarContinuous-v0")

    @app.route("/")
    def main():
        return render_template('index.html')
    #
    #
    #
    #
    #
    # API ROUTES
    @app.route("/getpair")
    def get_pair(env=None):
        if env is None:
            env = request.args.get('env')
        data = json.dumps(trajectory_builder.get_pair(env))
        return data

    @app.route('/preference', methods=['POST'])
    def update_text():
        user_pref = request.json
        env = user_pref["env"]
        del user_pref['env']

        pref_db = db_for_env[env]
        pref_db.append(user_pref)
        if len(pref_db) % 10 == 0:
            threading.Thread(target=save_pref_db(pref_db, env)).start()

        return get_pair(env)

    return app
