
from .open_api import register as register_open_api_routes
from .executions import register as register_executions_routes
from .tasks import register as register_tasks_routes

def register(app, rt, prefix=""):
    register_open_api_routes(app, rt, prefix)
    register_executions_routes(app, rt, prefix)
    register_tasks_routes(app, rt, prefix)

