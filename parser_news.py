import requests
from bs4 import BeautifulSoup
import time
import traceback
from datetime import datetime, timedelta
import threading
from queue import Queue
import re
from urllib.parse import urljoin, urlparse
import logging


class SmartNewsParser:
    def __init__(self):
        self.parsed_links = set()
        self.news_queue = Queue()
        self.is_running = False
        self.callbacks = []

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Расширенные конфигурации с приоритетом свежих новостей
        self.site_configs = {
            "lenta.ru": {
                "news_container": ".topnews__first-topic, .card-big, .card-mini, .item, .js-top-seven",
                "title": ".card-big__title, .card-mini__title, .item__title",
                "link": ".card-mini._topnews",
                "time": ".card-big__info, .card-mini__info",
                "base_url": "https://lenta.ru",
                "fresh_indicators": [
                    "только что",
                    "минуту",
                    "час",
                    "сегодня",
                    "сейчас",
                ],
            },
            "ria.ru": {
                "news_container": ".list-item, .cell-list__item, .news-item",
                "title": ".list-item__title, .cell-list__item-title, .news-item__title",
                "link": "a",
                "time": ".list-item__info, .cell-list__item-date, .news-item__date",
                "base_url": "https://ria.ru",
                "fresh_indicators": ["только что", "минуту", "час", "сегодня"],
            },
            "rbc.ru": {
                "news_container": ".news-feed__item, .js-news-feed-item, .main__feed__item",
                "title": ".news-feed__item__title",
                "link": "a.news-feed__item",
                "time": ".news-feed__item__date",
                "base_url": "https://www.rbc.ru",
                "fresh_indicators": ["только что", "минут", "час", "сегодня"],
            },
            "kommersant.ru": {
                "news_container": ".main__news, .news-card, .uho",
                "title": ".main__news__title, .news-card__title, .uho__link",
                "link": "a",
                "time": ".main__news__date, .news-card__date, .uho__time",
                "base_url": "https://www.kommersant.ru",
                "fresh_indicators": ["только что", "минут", "час", "сегодня"],
            },
            "tass.ru": {
                "news_container": ".MainBlock_grid__PquYX, .news-list__item, .content-item, .news-item",
                "title": ".MaterialCard_text___TyAy, .news-list__item-title, .content-item__title",
                "link": ".EnhancedLink_box__DuNJV",
                "time": ".Date_text__vYnLZ, .news-list__item-date, .content-item__date",
                "base_url": "https://tass.ru",
                "fresh_indicators": ["только что", "минут", "час", "сегодня"],
            },
        }

    def is_fresh_news(self, element, config):
        """Определяет, является ли новость свежей"""
        try:
            # Проверяем время публикации
            time_elem = element.select_one(config["time"])
            if time_elem:
                time_text = time_elem.get_text(strip=True).lower()

                # Ищем индикаторы свежести
                for indicator in config.get("fresh_indicators", []):
                    if indicator in time_text:
                        return True

                # Проверяем временные метки (минуты, часы)
                time_patterns = [
                    r"(\d+)\s*минут",  # 5 минут назад
                    r"(\d+)\s*час",  # 2 часа назад
                    r"только что",
                    r"сейчас",
                    r"сегодня",
                ]

                for pattern in time_patterns:
                    if re.search(pattern, time_text):
                        return True

            # Проверяем позицию в списке (первые элементы обычно свежие)
            parent = element.find_parent()
            if parent:
                siblings = parent.find_all(element.name, recursive=False)
                if siblings and siblings.index(element) < 10:  # Первые 10 элементов
                    return True

            return False

        except Exception as e:
            self.logger.debug(f"Error checking freshness: {e}")
            return True  # Если не можем определить, считаем свежей

    def parse_news_item(self, element, config, base_url):
        """Парсинг отдельного элемента новости с проверкой свежести"""
        try:
            # Извлекаем заголовок
            title_elem = element.select_one(config["title"])
            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            if not title or len(title) < 10:
                return None

            # Извлекаем ссылку
            link_elem = element.select_one(config["link"])
            if not link_elem:
                link_elem = title_elem.find("a")

            link = link_elem.get("href", "") if link_elem else ""
            if not link:
                return None

            if link and not link.startswith("http"):
                link = urljoin(config["base_url"], link)

            # Проверяем, не парсили ли мы уже эту ссылку
            if link in self.parsed_links:
                return None

            # Извлекаем время
            time_elem = element.select_one(config["time"])
            time_text = time_elem.get_text(strip=True) if time_elem else "Недавно"

            news_item = {
                "title": title,
                "link": link,
                "time": time_text,
                "timestamp": datetime.now().isoformat(),
                "domain": urlparse(base_url).netloc,
                "is_fresh": self.is_fresh_news(element, config),
            }

            return news_item

        except Exception as e:
            self.logger.debug(f"Error parsing news item: {e}")
            return None

    def get_fresh_news_sections(self, url):
        """Получает разделы со свежими новостями для каждого сайта"""
        domain = urlparse(url).netloc

        if "lenta.ru" in domain:
            return [
                "https://lenta.ru",  # Главная страница
                "https://lenta.ru/rubrics/russia/",  # Россия
                "https://lenta.ru/rubrics/world/",  # Мир
            ]
        elif "ria.ru" in domain:
            return ["https://ria.ru", "https://ria.ru/lenta/"]
        elif "rbc.ru" in domain:
            return ["https://www.rbc.ru/", "https://www.rbc.ru/short_news"]
        elif "kommersant.ru" in domain:
            return ["https://www.kommersant.ru/", "https://www.kommersant.ru/lenta"]
        elif "tass.ru" in domain:
            # return ["https://tass.ru/", "https://tass.ru/novosti-partnerov"]
            return ["https://tass.ru/"]
        else:
            return [url]

    def parse_site(self, url):
        """Парсинг свежих новостей с сайта"""
        config = self.get_site_config(url)
        if not config:
            self.logger.error(f"No configuration for site: {url}")
            return []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Ищем элементы новостей
            news_elements = []
            for selector in config["news_container"].split(", "):
                elements = soup.select(selector.strip())
                news_elements.extend(elements)

            if not news_elements:
                self.logger.warning(f"No news elements found for {url}")
                return []

            fresh_news = []
            for element in news_elements[:30]:  # Проверяем больше элементов
                news_item = self.parse_news_item(element, config, config["base_url"])
                if news_item and news_item["is_fresh"]:
                    self.parsed_links.add(news_item["link"])
                    fresh_news.append(news_item)
                    self.news_queue.put(news_item)
                    self.notify_callbacks(news_item)

            return fresh_news

        except Exception as e:
            self.logger.error(f"Error parsing {url}: {e}")
            return []

    def get_site_config(self, url):
        """Получить конфигурацию для сайта"""
        domain = urlparse(url).netloc.replace("www.", "")
        for site_domain, config in self.site_configs.items():
            if site_domain in domain:
                return config
        return None

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def notify_callbacks(self, news_item):
        for callback in self.callbacks:
            try:
                callback(news_item)
            except Exception as e:
                print(traceback.format_exc())
                self.logger.error(f"Callback error: {e}")

    def start_monitoring(self, urls, interval=30):
        """Запуск мониторинга с оптимизацией для свежих новостей"""
        self.is_running = True

        # Расширяем список URL для мониторинга
        expanded_urls = []
        for url in urls:
            expanded_urls.extend(self.get_fresh_news_sections(url))

        self.logger.info(f"Monitoring {len(expanded_urls)} news sections")

        def monitor_loop():
            while self.is_running:
                total_fresh = 0
                for url in expanded_urls:
                    print(f"Checking {url} for fresh news...")
                    try:
                        fresh_news = self.parse_site(url)
                        if fresh_news:
                            total_fresh += len(fresh_news)
                            self.logger.info(
                                f"Found {len(fresh_news)} fresh news from {urlparse(url).netloc}"
                            )
                    except Exception as e:
                        self.logger.error(f"Error monitoring {url}: {e}")

                if total_fresh > 0:
                    self.logger.info(f"Total fresh news found: {total_fresh}")

                time.sleep(interval)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        self.is_running = False
