import requests
import getpass
import re
import openai
 
USERNAME = input('USERNAME: ')
PASSWORD = getpass.getpass("PASSCODE: ")
 
response = requests.post(f'https://obsidian.ccs.ornl.gov/token', data={
    "username": USERNAME, 'password': PASSWORD,
})
TOKEN = response.json()['access_token']
 
with open(".nemotron_token", 'w') as file:
    file.write(TOKEN)
 
client = openai.OpenAI(
    base_url="https://obsidian.ccs.ornl.gov/ai/nemotron/api/v1",
    api_key=TOKEN,
)