from flask import Flask, render_template
from flask import request
from flask import jsonify
import json


def get_pref_db():
    try:
        f = open("pref_db.json")
        return json.load(f)
    except IOError:
        f = open("pref_db.json", "x")
        new_f = {"preferences": []}
        json.dump(new_f, f)
        return new_f
    finally:
        f.close()


def save_pref_db(pref_db):
    with open('pref_db.json', 'w') as json_file:
        json.dump(pref_db, json_file)


def get_webapp(trajectory_builder):
    app = Flask(__name__)

    @app.route("/")
    def main():
        return render_template('index.html')

    @app.route("/getpair")
    def get_pair():
        return trajectory_builder.get_pair()

    @app.route('/preference', methods=['POST'])
    def update_text():
        pref_db = get_pref_db()
        pref_db['preferences'].append(request.json)
        save_pref_db(pref_db)
        return jsonify(get_pair())

    return app
