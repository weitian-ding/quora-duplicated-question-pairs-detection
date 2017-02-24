#!/bin/bash

root_dir=GoC_weather_data

if [ $# -ne 5 ]; then
    echo "Usuage: $0 [[station_id]] [[year_begin]] [[year_end]] [[month_begin]] [[month_end]]"
    exit 1
fi


dir="${root_dir}/${1}"
rm -rf $dir # remove if it presents
mkdir -p $dir

for year in `seq ${2} ${3}`
do
	for month in `seq ${4} ${5}`
	do
		wget --content-disposition -O "${dir}/${year}-${month}.csv" "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=${1}&Year=${year}&Month=${month}&Day=14&timeframe=1&submit=Download+Data"
	done
done