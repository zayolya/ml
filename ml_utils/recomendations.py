import pandas as pd
from datetime import datetime
from schema import PostGet


# колонки для предсказания (они же были на обучении)
train_cols = ['gender', 'age', 'country', 'city', 'exp_group', 'topic', 'TextCluster', 'DistanceTo1Cluster',
              'DistanceTo2Cluster', 'DistanceTo3Cluster', 'DistanceTo4Cluster', 'DistanceTo5Cluster',
              'DistanceTo6Cluster', 'DistanceTo7Cluster', 'DistanceTo8Cluster', 'DistanceTo9Cluster',
              'DistanceTo10Cluster', 'DistanceTo11Cluster', 'DistanceTo12Cluster', 'hour', 'weekday', 'month']


def get_post_recommendations(date: datetime, limit: int, user_row: pd.DataFrame, posts: pd.DataFrame, model) -> list[PostGet]:
    """Рекомендовать пользователю посты, которые ему понравятся, на дату"""

    # Делаем таблицу из всех постов и пользователя
    df = pd.merge(user_row, posts, how='cross')

    # Добавляем дату к таблице, и отдельные свойства из даты
    df['timestamp'] = pd.to_datetime(date)
    df['hour'] = date.hour
    df['weekday'] = date.weekday()
    df['month'] = date.month

    # Получаем предсказания
    train_data = df[train_cols]
    probs_1 = model.predict_proba(train_data)[:, 1]
    df_probs = pd.DataFrame(probs_1, columns=['prob'])
    users_data = pd.concat([df, df_probs], axis=1)
    users_data.sort_values(by=['prob'], ascending=False, inplace=True)

    final_data = users_data.head(limit)

    return [PostGet(id=int(row['post_id']), text=row['text'], topic=row['topic']) for index, row in final_data.iterrows()]