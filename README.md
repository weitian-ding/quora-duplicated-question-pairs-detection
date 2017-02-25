# stat442-project

## Dataset construction

### Step 1: 
Download Images from [AMOS](http://amos.cse.wustl.edu/browse_with_filters).
```
python download_amos.py [[camera_id]] [[year_begin]] [[year_end]] [[month_begin]] [[month_end]]
```

### Step 2: 
Download Weather Conditions from [GoC historical weather databse](http://climate.weather.gc.ca/historical_data/search_historic_data_e.html).
```
./get-weather-cond.bash [[station_id]] [[year_begin]] [[year_end]] [[month_begin]] [[month_end]]
```

### Step 3: 
Filter images taken at night time and generate weather condition labels for all remaining images.
```
python weather-label-gen.py [[camera_id]] [[station_id]] [[local_timezone]]
```

