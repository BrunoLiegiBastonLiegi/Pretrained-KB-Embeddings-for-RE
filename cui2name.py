import requests, regex
from bs4 import BeautifulSoup

UMLS_key = '38791509-7b9c-4aac-a6a2-8728faf28bc1'

data = {'apikey': UMLS_key}
r = requests.post('https://utslogin.nlm.nih.gov/cas/v1/api-key', data=data)
soup = BeautifulSoup(r.text, "lxml")
#print(soup.prettify())

TGT = regex.search('TGT-(.+)-cas', soup.form['action']).group(0)
print('> Generated TGT:\n\t', TGT)

CUIs = ['C0009044', 'C0155502']

for c in CUIs:
    # request service ticket
    data = {'service': 'http://umlsks.nlm.nih.gov'}
    r = requests.post('https://utslogin.nlm.nih.gov/cas/v1/tickets/' + TGT, data=data)
    ticket = r.text
    #print(ticket)

    payload = {'ticket': ticket}
    r = requests.get('https://uts-ws.nlm.nih.gov/rest/content/current/CUI/' + c, params=payload)
    name = r.json()['result']['name']
    print(name)

