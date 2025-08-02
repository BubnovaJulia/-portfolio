from sqlalchemy import create_engine
from typing import List  
from fastapi import FastAPI, HTTPException  
from datetime import datetime  
import pandas as pd  
import os  
from catboost import CatBoostClassifier 
from pydantic import BaseModel  

app = FastAPI()  # Создаем экземпляр FastAPI

# Определяем модель данных для получения постов
class PostGet(BaseModel):
    id: int  # Идентификатор поста
    text: str  # Текст поста
    topic: str  # Тематика поста

    class Config:
        orm_mode = True  # Включаем режим ORM для работы с SQLAlchemy

# Функция для получения пути к модели в зависимости от окружения
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # Проверяем, выполняется ли код в LMS
        MODEL_PATH = '/workdir/user_input/model'  # Путь к модели в LMS
    else:
        MODEL_PATH = path  # Путь к модели в локальном окружении
    return MODEL_PATH

# Функция для загрузки данных из базы данных с использованием SQLAlchemy и pandas
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000  # Размер чанка для загрузки данных
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"  # Строка подключения к базе данных
    )
    conn = engine.connect().execution_options(stream_results=True)  # Устанавливаем соединение с БД
    chunks = []  # Список для хранения загруженных чанков
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):  # Загружаем данные по частям
        chunks.append(chunk_dataframe)  # Добавляем чанк в список
    conn.close()  # Закрываем соединение
    return pd.concat(chunks, ignore_index=True)  # Объединяем все чанки в один DataFrame

# Функция для загрузки данных из нескольких таблиц
def load_data():
    """Загрузка данных из БД"""
    User = batch_load_sql('SELECT * FROM j_bubnova_users_lesson_22 ')  # Загружаем данные пользователей
    Post = batch_load_sql('SELECT * FROM j_bubnova_posts_lesson_22 ')  # Загружаем данные постов
    Liked = batch_load_sql('SELECT * FROM j_bubnova_liked_lesson_22')  # Загружаем данные о лайках
    ordered_columns_post = ['post_id', 'text', 'topic']  # Задаем порядок колонок для постов
    post_df = Post[ordered_columns_post]  # Создаем DataFrame для постов
    ordered_columns_user = ['user_id', 'gender', 'city', 'exp_group', 'os', 'country', 'source', 'age']  # Задаем порядок колонок для пользователей
    user_df = User[ordered_columns_user]  # Создаем DataFrame для пользователей
    liked_df = Liked[['liked_posts', 'user_id']]  # Создаем DataFrame для лайков

    return post_df, user_df, liked_df  # Возвращаем загруженные DataFrame

# Функция для загрузки модели CatBoost
def load_models():
    model_path = get_model_path("catboost_model_new_1")  # Получаем путь к модели
    model = CatBoostClassifier()  # Создаем экземпляр классификатора CatBoost
    model.load_model(model_path)  # Загружаем модель из файла
    return model  # Возвращаем загруженную модель

# Функция для загрузки признаков для предсказания
def load_features(id: int, time: datetime, user_df: pd.DataFrame, post_df: pd.DataFrame, liked_df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = ['post_id', 'user_id', 'gender', 'age', 'country', 'city', 'exp_group',
                       'os', 'source', 'text', 'topic', 'year', 'month', 'day', 'hour']  # Задаем порядок ризнаков
    liked_posts = liked_df.loc[liked_df.user_id == id, 'liked_posts']  # Получаем посты, которые пользователь лайкнул
    if liked_posts.empty:
        liked_posts = []  # Если лайков нет, создаем пустой список
    else:
        liked_posts = liked_posts.values  # Иначе получаем лайкнутые посты
    post_df = post_df[~post_df['post_id'].isin(liked_posts)]  # Исключаем лайкнутые посты из DataFrame
    all_features_df = user_df.loc[user_df.user_id == id].merge(post_df, how='cross')  # Создаем DataFrame с признаками для пользователя
    all_features_df['year'] = time.year  # Добавляем год
    all_features_df['month'] = time.month  # Добавляем месяц
    all_features_df['day'] = time.day  # Добавляем день
    all_features_df['hour'] = time.hour  # Добавляем час
    all_features_df = all_features_df[ordered_columns]  # Упорядочиваем колонки

    return all_features_df  # Возвращаем DataFrame с признаками

loaded_catboost = load_models()  # Загружаем модель CatBoost
# Глобальные переменные для хранения загруженных данных
post_df, user_df, liked_df = load_data()  # Загружаем данные

# Функция для получения рекомендаций постов
def post_rec(id: int, time: datetime, limit: int) -> List[dict]:
    total_df = load_features(id, time, user_df, post_df, liked_df)  # Загружаем признаки
    total_df['preds_proba'] = loaded_catboost.predict_proba(total_df)[:, 1]  # Получаем вероятности предсказаний
    total_df.rename(columns={"post_id": "id"}, inplace=True)  # Переименовываем колонку post_id в id
    total_df = total_df.sort_values('preds_proba', ascending=False)[['id', 'text', 'topic']].head(limit)  # Сортируем и выбираем топ N постов

    return total_df.to_dict(orient='records')  # Возвращаем результаты в виде списка словарей

@app.get("/post/recommendations/", response_model=List[PostGet])  # Определяем маршрут для получения рекомендаций
def get_post_rec(id: int, time: str, limit: int = 5) -> List[PostGet]:
    time = datetime.fromisoformat(time)  # Преобразование строки в datetime
    return post_rec(id, time, limit)  # Возвращаем рекомендации

