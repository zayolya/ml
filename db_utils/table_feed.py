from sqlalchemy.orm import relationship

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey

from db_utils.database import Base
from db_utils.table_post import Post
from db_utils.table_user import User


class Feed(Base):
    __tablename__ = "feed_action"
    user_id = Column(Integer, ForeignKey(User.id), primary_key=True)
    post_id = Column(Integer, ForeignKey(Post.id), primary_key=True)
    action = Column(String, primary_key=True)
    time = Column(DateTime, primary_key=True)
    user = relationship('User')
    post = relationship('Post')
