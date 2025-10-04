from newspaper import Article

article = Article(
    "https://editorial.rbc.ru/mantr25s-a7-3lmn-m?from=newsfeed", language="ru"
)
article.download()
article.parse()

print(article.text)
