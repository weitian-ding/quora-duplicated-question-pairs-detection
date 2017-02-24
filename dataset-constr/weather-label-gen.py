#import arrow
import os
from datetime import tzinfo, datetime
from pytz import timezone
import csv

CSV_FILENAME = 'GoC_weather_data/7558/2017/1.csv'
HEADER_OFFSET = 16
WEATHER_TIMEZONE = 'Canada/Atlantic'
IMG_PATH = 'AMOS_Data/00017965/2017.01'
WEATHER_DEFAULT = 'DNE'
OUTPUT_FILENAME = 'labels.csv'

# converts a timestamp to integers
def utc_to_int(ts):
    ts_round = ts.replace(hour = ts.hour + 1 if (ts.minute > 30 and ts.hour < 23) else ts.hour)  # round to nearest hour
    return int(ts_round.strftime("%Y%m%d%H"))


def main():
    weather_table = {}
    # read the weather conditions
    with open(CSV_FILENAME, 'r') as in_file:
        # skip the headers
        for _ in range(HEADER_OFFSET):
            next(in_file)

        reader = csv.DictReader(in_file)

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

    # writes the labels to csv file
    with open(OUTPUT_FILENAME, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(labels)


if __name__ == "__main__":
    main()