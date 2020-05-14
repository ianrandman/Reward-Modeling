from flask import Flask, render_template
from flask import request
import json
import threading


"""
This file facilitates the Flask backend routes and the interfacing with the pref_db json files to save new preferences.
"""


def synchronized(func):
    """
    Allows for the synchronization of functions that interface with the json database files. Necessary when multiple
    users may be giving feedback at once, since reading and writing to the database can take some time.
    """
    func.__lock__ = threading.Lock()

    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)
    return synced_func


@synchronized
def begin_save(pref_db, env):
    threading.Thread(target=save_pref_db, args=[pref_db, env]).start()


@synchronized
def add_pref(pref, prefs):
    prefs.append(pref)

@synchronized
def save_pref_db(pref_db, env):
    """
    Saves the given pref_db list into the pref_db file for the specified environment.
    """
    with open("preferences/"+env+'/pref_db.json', 'r') as json_file:
        try:
            old_pref_db = json.load(json_file)
        except Exception as e:
            raise e
        with open("preferences/" + env + '/pref_db.json', 'w') as json_file:
            old_pref_db.extend(pref_db)
            json.dump(old_pref_db, json_file)


def get_pref_db(env):
    """
    Pulls the appropriate pref_db file for the specified environment. Returns as a python list of the db
    """
    with open("preferences/"+env+"/pref_db.json", 'r') as f:
        try:
            pref_db = json.load(f)
        except Exception as e:
            raise e
        return pref_db if len(pref_db) > 0 else None


def get_webapp(trajectory_builder, env_lst):
    app = Flask(__name__)

    # create a pref_db for all active environments
    db_for_env = {}
    for env in env_lst:
        pref_db = get_pref_db(env)
        if pref_db is None:
            pref_db = []
        db_for_env[env] = pref_db

    ####################################################################################
    # PAGE ROUTES

    # main page for giving feedback to the OpenAI Gym CartPole-v1 environment
    @app.route("/cartpole")
    def cartpole():
        return render_template('env.html', env_name="CartPole-v1")

    # main page for giving feedback to the OpenAI Gym MountainCarContinuous-v0 environment
    @app.route("/mountaincar")
    def mountaincar():
        return render_template('env.html', env_name="MountainCarContinuous-v0")

    # main page for giving feedback to the OpenAI Gym LunarLanderContinuous-v2 environment
    @app.route("/lunarlander")
    def lunar_lander():
        return render_template('env.html', env_name="LunarLanderContinuous-v2")

    # main page for giving feedback to the OpenAI Gym Pendulum-v0 environment
    @app.route("/pendulum")
    def pendulum():
        return render_template('env.html', env_name="Pendulum-v0")

    # main index page. This shows a list of all available environments for giving feedback
    @app.route("/")
    def main():
        return render_template('index.html')

    ####################################################################################
    # API ROUTES

    # called to generate a new pair for feedback
    @app.route("/getpair")
    def get_pair(env=None):
        if env is None:
            env = request.args.get('env')
        data = json.dumps(trajectory_builder.get_pair(env))
        return data

    # called when a new preference is submitted. Saves the preference to the appropriate db
    # and returns a new pair for feedback
    @app.route('/preference', methods=['POST'])
    def update_text():
        user_pref = request.json
        env = user_pref["env"]
        del user_pref['env']

        pref_db = db_for_env[env]
        pref_db.append(user_pref)
        if len(pref_db) % 3 == 0:
            begin_save(pref_db, env)
            db_for_env[env] = []

        return get_pair(env)

    return app
