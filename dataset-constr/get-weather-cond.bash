cd Goc_weather_data
for year in `seq 2016 2016`
do 
	for month in `seq 1 12`
	do
		wget --content-disposition "http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=${1}&Year=${year}&Month=${month}&Day=14&timeframe=1&submit=Download+Data" 
	done
done