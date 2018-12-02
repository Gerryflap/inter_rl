import json

from flask import Flask, request
from flask_restful import Api, Resource
import util


def get_server(trainer):
    # Start parameter trainer

    app = Flask(__name__)
    api = Api(app)

    class Test(Resource):
        def get(self):
            return "Hello world"

    class ExperienceReceiver(Resource):
        def post(self):
            experiences = request.form['experiences']
            experiences = json.loads(experiences)
            experiences = [
                (
                    util.np_from_dict(s),
                    a,
                    r,
                    util.np_from_dict(sp),
                    term
                )
                for s, a, r, sp, term in experiences
            ]
            experiences = list(experiences)
            trainer.add_experiences(experiences)

    class Model(Resource):
        def get(self):
            return trainer.params_to_json()

    api.add_resource(Test, '/test')
    api.add_resource(ExperienceReceiver, '/experience')
    api.add_resource(Model, '/model')

    return app


