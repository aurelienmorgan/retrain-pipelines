
from .open_api import register as register_open_api_routes
from .executions import register as register_executions_routes

def register(app, rt, prefix=""):
    register_open_api_routes(app, rt, prefix)
    register_executions_routes(app, rt, prefix)

