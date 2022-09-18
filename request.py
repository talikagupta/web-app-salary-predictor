import requests

# URL -- host it locally
url = 'http://localhost:5000/api/'

# value of the experience that you want to test
payload = {
	'exp':20
}

r = requests.post(url,json=payload)

print(r.json())
