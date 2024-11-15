from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="prompt page.html"
    )
@app.get("/r/")
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="result.html"
    )
@app.get("/f/")
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="feedback.html"
    )