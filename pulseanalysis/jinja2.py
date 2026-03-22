from django.conf import settings
from django.urls import NoReverseMatch, reverse
from jinja2 import Environment


def url_for(endpoint: str, **values):
    if endpoint == "static":
        filename = values.get("filename", "")
        return f"{settings.STATIC_URL}{filename}".replace("//", "/")
    try:
        return reverse(endpoint, kwargs=values)
    except NoReverseMatch:
        return "#"
def environment(**options):
    env = Environment(**options)
    env.globals.update(url_for=url_for)
    return env
