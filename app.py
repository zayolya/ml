from datetime import datetime
from http.client import HTTPException
from typing import List
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from db_utils.database import batch_load_sql, SessionLocal
from ml_utils.load_model import load_models
from ml_utils.recomendations import get_post_recommendations
from schema import PostGet, UserGet, FeedGet
from db_utils.table_feed import Feed
from db_utils.table_post import Post
from db_utils.table_user import User

app = FastAPI()
model = load_models()
USERS = batch_load_sql("SELECT * FROM public.user_data")
POSTS = batch_load_sql("SELECT * FROM osavinova_posts_lesson_22")


def get_db():
    with SessionLocal() as db:
        return db


@app.get('/post/recommendations/', response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    """Получить предсказания какие посты понравятся пользователю на дату"""
    user_row = USERS.loc[USERS['user_id'] == id]
    if user_row.shape[0] == 0:
        pass
    return get_post_recommendations(time, limit, user_row, POSTS, model)


@app.get('/user/{id}', response_model=UserGet)
def get_user(id: int, db: Session = Depends(get_db)) -> UserGet:
    user = db.query(User).filter(User.id == id).one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail='User not found')
    return user


@app.get('/post/{id}', response_model=PostGet)
def get_post(id: int, db: Session = Depends(get_db)) -> PostGet:
    post = db.query(Post).filter(Post.id == id).one_or_none()
    if post is None:
        raise HTTPException(status_code=404, detail='Post not found')
    return post


@app.get('/user/{id}/feed', response_model=List[FeedGet])
def get_feeds_by_user_id(id: int, db: Session = Depends(get_db), limit: int = 10) -> List[FeedGet]:
    return db.query(Feed).join(User).filter(User.id == id).order_by(Feed.time.desc()).limit(limit).all()


@app.get('/post/{id}/feed', response_model=List[FeedGet])
def get_feeds_by_post_id(id: int, db: Session = Depends(get_db), limit: int = 10) -> List[FeedGet]:
    return db.query(Feed).join(Post).filter(Post.id == id).order_by(Feed.time.desc()).limit(limit).all()
