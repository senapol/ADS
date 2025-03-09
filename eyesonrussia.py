import requests
import time
import json
import pandas as pd

def get_events_data():
    start_time = time.time()

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0', 'Accept':'*/*','Content-Type':'application/json'}  
    map_url = "https://eyesonrussia.org/events.geojson"
    response = requests.get(map_url, headers=headers).json()

    features = response['features']
    print(features[0])
    events = []

    for feature in features:
        try:
            properties = feature['properties']
            geometry = feature['geometry']
            event_data = {
                'id': properties.get('id', None),
                'date': properties.get('verifiedDate', '')[:10] if properties.get('verifiedDate') else None,
                'url': properties.get('url', None),
                'status': properties.get('status', None),
                'credit': properties.get('credit', None),
                'description': properties.get('description', None),
                'country': properties.get('country', None),
                'province': properties.get('province', None),
                'district': properties.get('district', None),
                'city': properties.get('city', None),
                'categories': json.dumps(properties.get('categories', [])),  # Convert list to JSON string
                'latitude': geometry['coordinates'][1] if geometry and 'coordinates' in geometry else None,
                'longitude': geometry['coordinates'][0] if geometry and 'coordinates' in geometry else None
            }

            events.append(event_data)
            print(f'Event {event_data["id"]} parsed successfully')
        except Exception as e:
            print(f'Error {e}: Could not parse event')

    print(f'Parsed {len(events)} events in {time.time() - start_time} seconds')

    # save events to file
    df = pd.DataFrame(events)
    df.to_csv('data/events.csv', index=False)

def main():
    # try:
    #     df = pd.read_csv('data/events.csv')
    #     print(df.head())
    #     print(df.columns)
    #     print()
    # except:
    #     print('File not found')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0', 'Accept':'*/*','Content-Type':'application/json'}  
    map_url = "https://eyesonrussia.org/events.geojson"
    response = requests.get(map_url, headers=headers).json()

    features = response['features']
    print(features[0]['properties'].keys())
    print(features[0]['geometry'].keys())
    for key in features[0]['properties'].keys():
        print(f'key: {key}: {features[0]["properties"][key]}')

    for key in features[0]['geometry'].keys():
        print(f'key: {key}: {features[0]["geometry"][key]}')

    categories = []
    descriptions = []
    for feature in features:
        category = feature['properties'].get('categories', [])
        description = feature['properties'].get('description', None)
        if len(category) > 1:
            for cat in category:
                if cat not in categories:
                    categories.append(cat)
        else:
            if category not in categories:
                categories.append(category)
        # if description not in descriptions:
        #     descriptions.append(description)

    print(f'Categories: {categories}')
    print(f'Number of categories: {len(categories)}')
    # print(f'Description: {descriptions}')
if __name__ == '__main__':
    main()