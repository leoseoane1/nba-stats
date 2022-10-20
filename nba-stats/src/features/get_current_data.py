import requests

url = "https://api-nba-v1.p.rapidapi.com/seasons"

headers = {
	"X-RapidAPI-Key": "91010406e1msh39fce9992657ffep18befajsnc3b7eab53714",
	"X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers)

print(response.text)