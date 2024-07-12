from sqlalchemy import Column, Integer, String, func, desc

from db_utils.database import Base, SessionLocal


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String)
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)


if __name__ == '__main__':
    session = SessionLocal()
    users = session.query(User.country, User.os, func.count(User.id).label('count')) \
        .filter(User.exp_group == 3)\
        .group_by(User.country, User.os)\
        .having(func.count(User.id) > 100)\
        .order_by(desc(func.count(User.id)))\
        .all()

    print([(user.country, user.os, user.count) for user in users])
