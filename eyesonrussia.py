import requests
import time
import json
import pandas as pd

def main():
    start_time = time.time()

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0', 'Accept':'*/*','Content-Type':'application/json'}  
    map_url = "https://eyesonrussia.org/events.geojson"
    response = requests.get(map_url, headers=headers).json()

    features = response['features']
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

            # event = Event(id, date, district, city, province, latitude, longitude, url, status, credit, categories)
            events.append(event_data)
            print(f'Event {event_data["id"]} parsed successfully')
        except Exception as e:
            print(f'Error {e}: Could not parse event')

    print(f'Parsed {len(events)} events in {time.time() - start_time} seconds')

    # Save events to file
    df = pd.DataFrame(events)
    df.to_csv('events.csv', index=False)


if __name__ == '__main__':
    main()