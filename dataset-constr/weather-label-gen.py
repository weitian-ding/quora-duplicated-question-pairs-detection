#import arrow
import os
from datetime import tzinfo, datetime
from pytz import timezone

CSV_FILENAME = 'GoC_weather_data/48569/2017/1.csv'
HEADER_OFFSET = 16
WEATHER_TIMEZONE = 'US/Eastern'
IMG_PATH = 'AMOS_Data/00017964/2017.01'
WEATHER_DEFAULT = 'DNE'

# converts a timestamp to integers
def utc_to_int(ts):
    ts_round = ts.replace(hour = ts.hour + 1 if (ts.minute > 30 and ts.hour < 23) else ts.hour)  # round to nearest hour
    return int(ts_round.strftime("%Y%m%d%H"))


def main():
    weather_table = {}
    # read the weather conditions
    import csv
    with open(CSV_FILENAME) as csvfile:
        # skip the headers
        for _ in range(HEADER_OFFSET):
            next(csvfile)

        reader = csv.DictReader(csvfile)

        for row in reader:
            ts = datetime.strptime(row['Date/Time'], '%Y-%m-%d %H:%M')
            ts = ts.replace(tzinfo=timezone(WEATHER_TIMEZONE))  # convert to local time
            ts = ts.astimezone(timezone('UTC'))  # convert to GMT
            ts_id = utc_to_int(ts)
            weather_table[ts_id] = row['Weather']
            print('read', ts, row['Weather'])

    images = os.listdir(IMG_PATH)
    labels = []
    for img in images:  # image filename is GMT timestamp
        img_ts = datetime.strptime(img, '%Y%m%d_%H%M%S.jpg')
        img_id = utc_to_int(img_ts)
        weather = weather_table.get(img_id, WEATHER_DEFAULT)
        labels.append(weather)  # look up the weather condition
        print("{} => {}".format(img, weather))


if __name__ == "__main__":
    main()