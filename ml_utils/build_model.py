import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

USER_QUERY = "SELECT * FROM public.user_data"
POST_QUERY = "SELECT * FROM public.post_text_df"
FEED_QUERY = "SELECT * FROM public.feed_data WHERE action='view' LIMIT 5000000"
DB_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
ENGINE = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
)


def plot_feature_importance(importance, names, model_type):
    """Визуализация важности признаков"""
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


if __name__ == '__main__':
    user_data = pd.read_sql(USER_QUERY, DB_URL)
    post_data = pd.read_sql(POST_QUERY, DB_URL)
    feed_data = pd.read_sql(FEED_QUERY, DB_URL)


    """EDA user"""
    # user_data.shape
    # user_data.head(5)
    # user_data['country'].value_counts()
    # user_data.info()
    # user_data.isna().sum()

    """EDA post"""
    # post_data.shape
    # post_data.head(5)
    # post_data.info()
    # post_data.describe()

    """EDA feed"""
    # feed_data.shape
    # feed_data.head(5)
    # feed_data.info()

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
    dists_columns = [f'DistanceTo{i+1}Cluster' for i in range(n_clusters)]
    kmeans_df = pd.DataFrame(data=kmeans.transform(pca_df), columns=dists_columns)
    kmeans_df.head()

    # присоединим новые фитчи к таблице постов
    new_posts = pd.concat([post_data, kmeans_df], axis=1)

    # проверка корреляции
    # corr = new_posts.select_dtypes(include=np.number).corr()
    # corr.style.background_gradient(cmap='coolwarm')

    # сохраним обработанные признаки
    new_posts.to_sql("osavinova_posts_lesson_22", con=ENGINE, if_exists='replace')

    # объеденим таблицы
    df = pd.merge(feed_data, new_posts, on='post_id', how='inner')
    df = pd.merge(user_data, df, on='user_id', how='inner')

    # выделим новые признаки из времени
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    df = df.set_index(['user_id', 'post_id'])

    # проверка корреляции
    # corr = df.select_dtypes(include=np.number).corr()
    # corr.style.background_gradient(cmap='coolwarm')

    # разделим выборку
    df_train = df[df.timestamp < '2021-12-15']
    df_test = df[df.timestamp >= '2021-12-15']

    X_tr = df_train.drop(['timestamp', 'text', 'action', 'target'], axis=1)
    y_tr = df_train['target']

    X_test = df_test.drop(['timestamp', 'text', 'action', 'target'], axis=1)
    y_test = df_test['target']

    object_cols = ['topic', 'TextCluster', 'gender', 'country', 'city', 'exp_group', 'hour', 'month', 'weekday', 'os',
                   'source']

    """Поиск наилучшей модели, занимает более 3х часов"""
    # splitter = TimeSeriesSplit(n_splits=4)
    #
    # params_grid = {'depth': [i for i in range(2, 7)], 'iterations': [100, 300], 'l2_leaf_reg': [5, 10, 15]}
    # grid_search = GridSearchCV(CatBoostClassifier(cat_features=object_cols), params_grid, scoring='roc_auc', cv=splitter)
    # grid_search.fit(X_tr, y_tr)
    #
    # best_model = grid_search.best_estimator_
    # predict_proba_test = best_model.predict_proba(X_test)[:, 1]
    # predict_proba_train = best_model.predict_proba(X_tr)[:, 1]
    # print(f'Best model ROC_AUC = {roc_auc_score(y_test, predict_proba_test)}')
    # print(f'Best model ROC_AUC train = {roc_auc_score(y_tr, predict_proba_train)}')
    # print(f'Best model params: {grid_search.best_params_}')

    # Лучшая моодель: ROC_AUC test = 0.654, ROC_AUC train = 0.689, model params: {'depth': 6, 'iterations': 300, 'l2_leaf_reg': 10}"""

    # catboost = CatBoostClassifier(learning_rate=0.2)  Качество на трейне: 0.724 Качество на тесте: 0.684
    # catboost = CatBoostClassifier() Качество на трейне: 0.728  Качество на тесте: 0.686"""
    # catboost = CatBoostClassifier(depth=6, iterations=400, l2_leaf_reg=10) Качество на трейне: 0.692 Качество на тесте: 0.659

    catboost = CatBoostClassifier(iterations=500)    # Качество на трейне: 0.724 Качество на тесте: 0.683
    catboost.fit(X_tr, y_tr, object_cols, logging_level='Verbose')
    print(f"Качество на трейне: {roc_auc_score(y_tr, catboost.predict_proba(X_tr)[:, 1])}")
    print(f"Качество на тесте: {roc_auc_score(y_test, catboost.predict_proba(X_test)[:, 1])}")

    # сохраним модель
    catboost.save_model('catboost_model', format="cbm")

    # сохраним столбцы обучения
    print("train columns", X_tr.columns)

    """train columns Index(['gender', 'age', 'country', 'city', 'exp_group', 'os', 'source',
       'topic', 'TextCluster', 'DistanceTo1Cluster', 'DistanceTo2Cluster',
       'DistanceTo3Cluster', 'DistanceTo4Cluster', 'DistanceTo5Cluster',
       'DistanceTo6Cluster', 'DistanceTo7Cluster', 'DistanceTo8Cluster',
       'DistanceTo9Cluster', 'DistanceTo10Cluster', 'DistanceTo11Cluster',
       'DistanceTo12Cluster', 'hour', 'weekday', 'month']"""