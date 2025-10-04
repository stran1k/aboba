from flask import Flask, render_template, request, redirect, url_for, session, flash
from asgiref.wsgi import WsgiToAsgi
import sqlite3, datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey123"


def get_connection():
    return sqlite3.connect("database.db", check_same_thread=False, timeout=30)


def get_news():
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    news = conn.execute("SELECT * FROM news ORDER BY time DESC;").fetchall()
    conn.close()
    return news


def get_all_genres():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT genre FROM news")
    result = cursor.fetchall()
    conn.close()

    genres = [g[0].strip() for g in result if g[0]]
    corrections = {
        "Присшествия": "Происшествия",
        "Социально-политическая ситуация": "Политика",
    }
    cleaned_genres = []
    for g in genres:
        if len(g) > 30 or ":" in g or "\n" in g:
            continue
        g = corrections.get(g, g)
        cleaned_genres.append(g)
    return sorted(set(cleaned_genres))

def add_bookmark(user_id, news_id):
    conn = get_connection()
    conn.execute(
        "INSERT INTO bookmarks (user_id, news_id, added_at) VALUES (?, ?, ?)",
        (user_id, news_id, int(datetime.datetime.now().timestamp()))
    )
    conn.commit()
    conn.close()


def remove_bookmark(user_id, news_id):
    conn = get_connection()
    conn.execute(
        "DELETE FROM bookmarks WHERE user_id=? AND news_id=?", 
        (user_id, news_id)
    )
    conn.commit()
    conn.close()


def get_user_bookmarks(user_id):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT news.* FROM bookmarks
        JOIN news ON bookmarks.news_id = news.id
        WHERE bookmarks.user_id=?
        ORDER BY bookmarks.added_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return rows


def is_bookmarked(user_id, news_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM bookmarks WHERE user_id=? AND news_id=?", 
        (user_id, news_id)
    ).fetchone()
    conn.close()
    return row is not None

@app.template_filter("format_time")
def format_time_filter(ts):
    try:
        return datetime.datetime.fromtimestamp(int(ts)).strftime("%H:%M %d.%m.%y")
    except Exception:
        return ts


@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    news_list = get_news()
    genres = get_all_genres()
    all_genres = [
        "Политика",
        "Экономика",
        "Происшествия",
        "Спорт",
        "Культура",
        "Наука",
        "Технологии",
        "Здоровье",
        "Путешествия",
        "Образование",
    ]

    conn = get_connection()
    user = conn.execute("SELECT id FROM users WHERE username=?", (session["user"],)).fetchone()
    conn.close()
    if user:
        bookmarked_ids = [row["id"] for row in get_user_bookmarks(user[0])]
    else:
        bookmarked_ids = []


    return render_template(
        "index.html",
        news=news_list,
        genres=genres,
        all_genres=all_genres,
        user=session["user"],
        bookmarked_ids=bookmarked_ids
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)
        try:
            conn = get_connection()
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)"
            )
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed_password),
            )
            conn.commit()
            conn.close()
            session["user"] = username
            return redirect(url_for("index"))
        except sqlite3.IntegrityError:
            flash("Такой пользователь уже существует", "danger")
    return render_template("auth.html", mode="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user"] = username
            return redirect(url_for("index"))
        else:
            flash("Неверный логин или пароль", "danger")
    return render_template("auth.html", mode="login")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Вы вышли из аккаунта", "info")
    return redirect(url_for("login"))
@app.route("/bookmark/add/<int:news_id>", methods=["POST"])
def bookmark_add(news_id):
    if "user" not in session:
        return redirect(url_for("login"))
    
    conn = get_connection()
    user = conn.execute("SELECT id FROM users WHERE username=?", (session["user"],)).fetchone()
    conn.close()

    if user:
        add_bookmark(user[0], news_id)
    return redirect(url_for("index"))


@app.route("/bookmark/remove/<int:news_id>", methods=["POST"])
def bookmark_remove(news_id):
    if "user" not in session:
        return redirect(url_for("login"))
    
    conn = get_connection()
    user = conn.execute("SELECT id FROM users WHERE username=?", (session["user"],)).fetchone()
    conn.close()

    if user:
        remove_bookmark(user[0], news_id)
    return redirect(url_for("index"))


app = WsgiToAsgi(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
