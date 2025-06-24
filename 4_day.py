import requests

def weather_details(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()


        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        wind_speed = data["wind"]["speed"]
        description = data["weather"][0]["description"]
        country = data["sys"]["country"]


        print(f" Weather Details in {city}, {country}")
        print(f"Temperature     : {temperature} °C")
        print(f"Feels Like      : {feels_like} °C")
        print(f"Humidity        : {humidity}%")
        print(f"Pressure        : {pressure} hPa")
        print(f"Wind Speed      : {wind_speed} m/s")
        print(f"Description     : {description}")


    except requests.exceptions.RequestException as e:
        print(f" Error fetching data: {e}")
    except KeyError:
        print(" Some data is missing in the response. Please check the city name or try again.")

city = input("Enter the city name: ")
api_key = "a9c05b58a9f0be9c1b9a4a54b73be893"
weather_details(city, api_key)
