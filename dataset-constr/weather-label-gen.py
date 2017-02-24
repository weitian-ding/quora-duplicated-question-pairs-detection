#import arrow
import os
from datetime import tzinfo, datetime
from pytz import timezone
import csv
import sys

CSV_ROOT = 'GoC_weather_data'
IMG_ROOT = 'AMOS_Data'

HEADER_OFFSET = 16
WEATHER_DEFAULT = 'DNE'
LABEL_FILENAME = 'labels_{}.csv'
MAPPING_FILENAME = 'mappings_{}.csv'

USAGE = '{} [[camera_id]] [[station_id]] [[local_timezone]]'

TIME_LOWER_BOUND = 9   # 9 a.m.
TIME_UPPER_BOUND = 16  # 4 p.m.

# converts a timestamp to integers
def utc_to_int(ts):
    ts_round = ts.replace(hour = ts.hour + 1 if (ts.minute > 30 and ts.hour < 23) else ts.hour)  # round to nearest hour
    return int(ts_round.strftime("%Y%m%d%H"))


# builds the weather condition lookup table
def build_weather_table(station_id, tz, table):
    mapping = []
    for spreadsheet in os.listdir("{}/{}".format(CSV_ROOT, station_id)):
        csv_filename = "{}/{}/{}".format(CSV_ROOT, station_id, spreadsheet)
        with open(csv_filename, 'r') as in_file:
            # skip the headers
            for _ in range(HEADER_OFFSET):
                next(in_file)

            reader = csv.DictReader(in_file)

            for row in reader:
                ts = datetime.strptime(row['Date/Time'], '%Y-%m-%d %H:%M')
                ts = ts.replace(tzinfo=timezone(tz))  # convert to local time
                ts = ts.astimezone(timezone('UTC'))  # convert to GMT
                ts_id = utc_to_int(ts)
                try:
                    weather = row['Weather'].split(',')[0]  # takes the first weather condition
                except AttributeError:
                    print("{} malformed".format(row))
                if weather not in mapping:
                    mapping.append(weather)
                table[ts_id] = mapping.index(weather)
                print('read', ts, row['Weather'])

    return mapping


# filters night-time images and generates weather condition labels
def filter_and_gen_labels(camera_id, tz, labels, weather_table):
    images = os.listdir("{}/{}".format(IMG_ROOT, camera_id))
    for img in images:  # image filename is GMT timestamp
        img_ts = datetime.strptime(img, '%Y%m%d_%H%M%S.jpg')
        img_ts = img_ts.replace(tzinfo=timezone('UTC'))

        img_path = "{}/{}/{}".format(IMG_ROOT, camera_id, img)

        # delete night time pictures
        img_local_hours = img_ts.astimezone(timezone(tz)).hour

        if img_local_hours < TIME_LOWER_BOUND or img_local_hours > TIME_UPPER_BOUND:
            os.remove(img_path)
            print("deleted {} due to night time".format(img_path))
        else:
            img_id = utc_to_int(img_ts)
            if img_id in weather_table:
                weather = weather_table[img_id]
            else:
                os.remove("{}/{}/{}".format(IMG_ROOT, camera_id, img))
                print('deleted {} due to missing weather condition'.format(img_path))
            labels.append(weather)  # look up the weather condition
            print("{} => {}".format(img, weather))


def main():
    if len(sys.argv) != 4:
        print(USAGE.format(sys.argv[0]))

    camera_id = sys.argv[1]
    station_id = sys.argv[2]
    tz = sys.argv[3]

    weather_table = {}
    labels = []

    mapping = build_weather_table(station_id, tz, weather_table)
    filter_and_gen_labels(camera_id, tz, labels, weather_table)

    # writes the labels to csv file
    with open(LABEL_FILENAME.format(camera_id), 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(labels)

    # writes mapping to csv file
    with open(MAPPING_FILENAME.format(station_id), 'w') as out_file:
        for idx,weather in enumerate(mapping):
            out_file.write("{},{}\n".format(idx, weather))


if __name__ == "__main__":
    main()