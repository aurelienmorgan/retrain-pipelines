
def get_cookie(req, key, default=None):
    return req.cookies.get(key, default)

def set_cookie(resp, key, value, path="/", max_age=86400 * 365):
    resp.set_cookie(key, value, path=path, max_age=max_age)

def get_ui_state(req, view, key, default=None):
    return get_cookie(req, f"{view}:{key}", default)

def set_ui_state(resp, view, key, value):
    set_cookie(resp, f"{view}:{key}", value)

