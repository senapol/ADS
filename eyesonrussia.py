import requests
import time

class Event:
    def __init__(self, id, date, district, city, province, latitude, longitude, source, status, credit, categories):
        self.id = id
        self.date = date
        self.district = district
        self.city = city
        self.province = province
        self.latitude = latitude
        self.longitude = longitude
        self.source = source
        self.status = status
        self.credit = credit
        self.categories = categories

    def __str__(self):
        return f'{self.date} - {self.city}, {self.province}'

def main():
    start_time = time.time()

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0', 'Accept':'*/*','Content-Type':'application/json'}  
    map_url = "https://eyesonrussia.org/events.geojson"
    response = requests.get(map_url, headers=headers).json()

    features = response['features']
    events = []

    for feature in features:
        try:
            id = feature['properties']['id']
            date = feature['properties']['verifiedDate'][:10]
            url = feature['properties']['url']
            status = feature['properties']['status']
            credit = feature['properties']['credit']
            description = feature['properties']['description']
            country = feature['properties']['country']
            province = feature['properties']['province']
            district = feature['properties']['district']
            city = feature['properties']['city']
            latitude = feature['geometry']['coordinates'][0]
            longitude = feature['geometry']['coordinates'][1]
            categories = feature['properties']['categories'][:]

            event = Event(id, date, district, city, province, latitude, longitude, url, status, credit, categories)
            events.append(event)
            print(f'Event {id} parsed successfully')
        except Exception as e:
            print(f'Error {e}: Could not parse event')

    print(f'Parsed {len(events)} events in {time.time() - start_time} seconds')

if __name__ == '__main__':
    main()