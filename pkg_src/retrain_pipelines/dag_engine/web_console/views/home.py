from fasthtml.common import Titled, P

def register(app, rt, prefix=""):
    @rt(f"{prefix}/")
    def hello():
        return Titled("Hello", P("Hello, World!"))

