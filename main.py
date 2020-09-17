from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
from utils import *

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request):
	# click on category
	categories, idxs2cat, cat2idxs = list_category()
	return templates.TemplateResponse("recommend.html", {"request": request, "category": categories, "values": cat2idxs})

@app.get("/categories/{cat_id}", response_class=HTMLResponse)
async def recommend_cat(request: Request, cat_id: int):
	categories, idxs2cat, cat2idxs = list_category()
	cat = idxs2cat[cat_id]
	cat_articles = recommend_articles(cat)
	return templates.TemplateResponse("recommend_articles.html", {"request": request, "category": cat, "articles": cat_articles[:10], "cat_id": cat_id})

@app.get("/articles/{cat_id}/{art_id}", response_class=HTMLResponse)
async def recommend_art(request: Request, art_id: int, cat_id: int):
	categories, idxs2cat, cat2idxs = list_category()
	cat = idxs2cat[cat_id]
	similar_df = tfidf_based_model(art_id, 11)
	headlines = similar_df.headline.tolist()
	return templates.TemplateResponse("recommend_articles_1.html", {"request": request, "category": cat, "articles": headlines[:10]})

@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})