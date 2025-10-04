import traceback
from parser_news import SmartNewsParser
from threading import Thread
import sqlite3
import uvicorn
import time
import subprocess
import asyncio
from newspaper import Article
from queue import Queue
from parser.newsParser import get_news
from urllib.parse import urlparse
import datetime

write_queue = Queue()

NEWS_EDITOR = "Напиши новость строго на русском языке. Обязательное условие: окончательный текст новости должен содержать не более 2 предложений. Передавай только факты, точно и по сути, без воды и лишних слов. Каждое предложение должно быть информативным. В ответе предоставь только текст новости, без лишних слов. Если информация слишком длинная, сокращай её, чтобы обязательно уложиться в 2 предложения.\nВот текст новости: "
GET_GENRE = "Определи жанр новости: политика, экономика, происшествия, спорт, культура, наука, технологии, здоровье, путешествия, образование. Пиши только один жанр строго на русском языке, без английских букв и транслита (транслит преобразуй в русский). Если жанр определить нельзя — пиши ‘другое’. В ответе предоставь только жанр, без лишних слов и объяснений. \nВот текст новости: "


def run_llama_server():
    subprocess.run(
        ["ollama", "serve"],
        # stdout=subprocess.DEVNULL,  # перенаправляем стандартный вывод
        # stderr=subprocess.DEVNULL,  # перенаправляем ошибки
    )


def run_llama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3:8b"],
        input=prompt,  # строка!
        capture_output=True,
        text=True,
    )
    return result.stdout


def get_connection():
    """Создаём новое соединение к БД"""
    return sqlite3.connect("database.db", check_same_thread=False, timeout=30)


def init_tables():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """
    )

    cursor.execute(
        """
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    domain TEXT NOT NULL,
    link TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    text TEXT DEFAULT NULL,
    genre TEXT DEFAULT NULL
)
"""
    )
    conn.commit()
    conn.close()


def check_news_without_text():
    """Проверяем наличие новостей без текста и удаляем их"""
    while True:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")

        all_news = cursor.execute(
            "SELECT * FROM news WHERE text IS NULL ORDER BY time ASC"
        ).fetchall()

        for news in all_news:
            (
                news_id,
                news_time,
                news_domain,
                news_link,
                news_title,
                news_text,
                news_genre,
            ) = news
            if news_domain == "editorial.rbc.ru":
                continue
            article = Article(news_link, language="ru")
            try:
                article.download()
                article.parse()
            except Exception:
                continue

            text = run_llama(prompt=f"{NEWS_EDITOR}{article.text}")
            genre = (
                run_llama(prompt=f"{GET_GENRE}{article.text}")
                .capitalize()
                .replace(".", "")
            )

            cursor.execute(
                "UPDATE news SET text = ?, genre = ? WHERE id = ?",
                (text.strip(), genre.strip(), news_id),
            )

            conn.commit()

            time.sleep(0.5)

        conn.commit()
        conn.close()

        time.sleep(30)


def cleanup_old_news():
    """Удаляет записи старше 3х часов по timestamp"""
    while True:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")

            # текущий timestamp
            now = int(time.time())
            three_hours_ago = now - 3 * 3600

            cursor.execute(
                "DELETE FROM news WHERE CAST(time AS INTEGER) < ?", (three_hours_ago,)
            )
            deleted = cursor.rowcount

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Ошибка очистки новостей: {e}")

        time.sleep(300)  # каждые 5 минут


def db_writer():
    """Поток для записи в базу данных"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    while True:
        task = write_queue.get()
        if task is None:
            break
        query, params = task
        try:
            cursor.execute(query, params)
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Ошибка записи: {e}")
        write_queue.task_done()
    conn.close()


async def main():
    init_tables()
    while True:
        try:
            all_news = await get_news()
            for news in all_news:
                news_time, news_title, news_link = news

                domain = urlparse(news_link).netloc

                query = """
                INSERT OR IGNORE INTO news (time, domain, link, title)
                VALUES (?, ?, ?, ?)
                """
                params = (str(news_time), domain, news_link, news_title)
                write_queue.put((query, params))
            await asyncio.sleep(60)
        except Exception:
            print(traceback.format_exc())
            await asyncio.sleep(60)
            continue


def start_uvicorn():
    uvicorn.run("app:app")


if __name__ == "__main__":
    # t2 = Thread(target=run_llama_server)
    # t2.start()
    # t3 = Thread(target=check_news_without_text)
    # t3.start()
    # writer_thread = Thread(target=db_writer, daemon=True)
    # writer_thread.start()
    # deleter = Thread(target=cleanup_old_news)
    # deleter.start()
    uv = Thread(target=start_uvicorn)
    uv.start()
    # asyncio.run(main())
