# stat442-project

## Dataset construction

### Step 1: Download Images 
```
python download_amos.py [[camera_id]] [[year_begin]] [[year_end]] [[month_begin]] [[month_end]]
```

### Step 2: Get Weather Conditions 
```
./get-weather-cond.bash [[station_id]] [[year_begin]] [[year_end]] [[month_begin]] [[month_end]]
```

### Step 3: Filter night time images and generate weather condition labels for all images
```
python weather-label-gen.py [[camera_id]] [[station_id]] [[local_timezone]]
```

