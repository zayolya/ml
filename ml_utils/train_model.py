import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine, text
from clearml import Task

USER_QUERY = "SELECT * FROM public.user_data"
POST_QUERY = "SELECT * FROM public.post_text_df"
FEED_QUERY = "SELECT * FROM public.feed_data WHERE action='view' LIMIT 5000000"
DB_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
ENGINE = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)


def load_tables():
    """Загрузка таблиц user_data, post_text_df, feed_data из БД"""
    engine_cloud = create_engine(DB_URL)
    user_data = pd.DataFrame(engine_cloud.connect().execute(text(USER_QUERY)))
    post_data = pd.DataFrame(engine_cloud.connect().execute(text(POST_QUERY)))
    feed_data = pd.DataFrame(engine_cloud.connect().execute(text(FEED_QUERY)))
    return (user_data, post_data, feed_data)


def process_posts(post_data: pd.DataFrame) -> pd.DataFrame:
    """Преобразование текста из столбца text в вещественные фитчи с помощью TFIDF, PCA, KMEANS"""
    # преобразуем столбец text в фитчи
    tfidf = TfidfVectorizer()
    text_matrix = tfidf.fit_transform(post_data['text'])
    texts_df = pd.DataFrame(text_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # понизим пространство с помощью pca
    m = 20
    pca = PCA(n_components=m)
    pca = pca.fit_transform(texts_df)
    pca_df = pd.DataFrame(data=pca, columns=[f'text_{i}' for i in range(m)])

    # сформируем кластеры из текстовых фитчей
    n_clusters = 12
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_df)
    post_data['TextCluster'] = kmeans.labels_
    dists_columns = [f'DistanceTo{i + 1}Cluster' for i in range(n_clusters)]
    kmeans_df = pd.DataFrame(data=kmeans.transform(pca_df), columns=dists_columns)
    kmeans_df.head()

    # присоединим новые фитчи к таблице постов
    new_posts = pd.concat([post_data, kmeans_df], axis=1)

    return new_posts


if __name__ == '__main__':
    user_data, post_data, feed_data = load_tables()

    new_posts = process_posts(post_data)
    # сохраним обработанные признаки
    new_posts.to_sql("osavinova_posts_final", con=ENGINE, if_exists='replace')

    # объеденим таблицы
    df = pd.merge(feed_data, new_posts, on='post_id', how='inner')
    df = pd.merge(user_data, df, on='user_id', how='inner')

    # удалим столбцы не участвующие в обучении
    df = df.drop(['os', 'source', 'action', 'text'], axis=1)

    df['exp_group'] = df['exp_group'].astype(object)
    df['gender'] = df['gender'].astype(object)

    # выделим новые признаки из времени
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    df = df.set_index(['user_id', 'post_id'])

    # разделим выборку
    df_train = df[df.timestamp < '2021-12-15']
    df_test = df[df.timestamp >= '2021-12-15']

    X_train = df_train.drop(['timestamp', 'target'], axis=1)
    y_train = df_train['target']

    X_test = df_test.drop(['timestamp', 'target'], axis=1)
    y_test = df_test['target']

    object_cols = ['topic', 'TextCluster', 'gender', 'country', 'city', 'exp_group', 'hour', 'month', 'weekday']

    # Вариации параметров для обучения
    train_params = [
        # {},
        # {'random_seed': 63, 'learning_rate': 0.5, 'early_stopping_rounds': 80},
        # {'depth': 6, 'l2_leaf_reg': 8, 'early_stopping_rounds': 70},
        {'learning_rate': 0.2, 'early_stopping_rounds': 60},
        {'learning_rate': 0.1, 'early_stopping_rounds': 50}
                    ]
    # Обучение с различными параметрами с отслеживанием в ClearML
    for i, param in enumerate(train_params):
        i += 3
        task = Task.init(project_name='Olgas catboost prj', task_name=f'Catboost task {i}', tags=['catboost'],
                         output_uri=True)
        # Log params
        task.connect(param)

        catboost = CatBoostClassifier(**param)
        catboost.fit(X_train, y_train, cat_features=object_cols, logging_level='Verbose')

        task.connect(catboost, name='catboost')

        roc_auc_train = roc_auc_score(y_train, catboost.predict_proba(X_train)[:, 1])
        roc_auc_test = roc_auc_score(y_test, catboost.predict_proba(X_test)[:, 1])
        print(f"Качество на трейне: {roc_auc_train}")
        print(f"Качество на тесте: {roc_auc_test}")

        logger = task.get_logger()
        logger.report_single_value('roc_auc_score на трейне:', roc_auc_train)
        logger.report_single_value('roc_auc_score на тесте:', roc_auc_test)

        # сохраним модель
        catboost.save_model(f'catboost_model_{i}', format="cbm")

        # сохраним столбцы обучения
        print("train columns", X_train.columns)

        joblib.dump(catboost, f"cat_boost_{i}", compress=True)

        task.close()
