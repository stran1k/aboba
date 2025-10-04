import datetime

timestamp = 1759471020
dt = datetime.datetime.fromtimestamp(timestamp)

print(datetime.datetime.fromtimestamp(timestamp).strftime("%d.%m.%Y %H:%M"))
