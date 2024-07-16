from datetime import datetime
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse

from db_utils.database import batch_load_sql
from ml_utils.load_model import load_models
from ml_utils.recomendations import get_post_recommendations
from schema import PostGet

app = FastAPI()
model = load_models()
USERS = batch_load_sql("SELECT * FROM public.user_data")
POSTS = batch_load_sql("SELECT * FROM osavinova_posts_final")


@app.get('/post/recommendations/', response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    """Получить предсказания какие посты понравятся пользователю на дату"""
    user_row = USERS.loc[USERS['user_id'] == id]
    if user_row.shape[0] == 0:
        raise HTTPException(status_code=404, detail='User not found')
    return get_post_recommendations(time, limit, user_row, POSTS, model)


@app.get("/liveness")
async def liveness():
    return JSONResponse(content={'status': 'ok'}, status_code=200)


@app.get("/readiness")
async def readiness():
    if model:
        return JSONResponse(content={'status': 'ok'}, status_code=200)
    else:
        return JSONResponse(content={"status": "error", "reason": "model"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)