import requests, regex, json, argparse
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt

# Arguments parser
parser = argparse.ArgumentParser(description='Convert the Knowledge Graph from CUIs to names.')
parser.add_argument('input_graph', help='Path to input graph.json file.')
args = parser.parse_args()

UMLS_key = '38791509-7b9c-4aac-a6a2-8728faf28bc1'

data = {'apikey': UMLS_key}
r = requests.post('https://utslogin.nlm.nih.gov/cas/v1/api-key', data=data)
soup = BeautifulSoup(r.text, "lxml")
#print(soup.prettify())

TGT = regex.search('TGT-(.+)-cas', soup.form['action']).group(0)
print('> Generated TGT:\n\t', TGT)

with open(args.input_graph) as f:
    g = json.load(f)

try:
    with open('cui2name.json') as f:
        cui2name = json.load(f)
except:
    cui2name = {}
    
kg = nx.Graph()
color_map = {}

for l in g['links']:

    s = regex.split('-', l['source'])
    t = regex.split('-', l['target'])

    for k in (s,t):
        for i, c in enumerate(k):
            try:
                k[i] = cui2name[c]
            except:
                # request service ticket
                data = {'service': 'http://umlsks.nlm.nih.gov'}
                r = requests.post('https://utslogin.nlm.nih.gov/cas/v1/tickets/' + TGT, data=data)
                ticket = r.text
                #print(ticket)

                # find CUI definition
                payload = {'ticket': ticket}
                r = requests.get('https://uts-ws.nlm.nih.gov/rest/content/current/CUI/' + c, params=payload)
                name = r.json()['result']['name']
                cui2name[c] = name
                k[i] = name
                print(c, ':', name)

    s = '-'.join(s)
    t = '-'.join(t)
    color_map[s] = 'orange'
    color_map[t] = 'cyan'
    kg.add_edge(s, t)

    
#print(cui2name)
kg = kg.subgraph(sorted(nx.connected_components(kg), key = len, reverse=True)[0])
nx.draw(kg, with_labels=True, node_color=[color_map[n] for n in kg.nodes()])
plt.show()

with open('cui2name.json', 'w') as f:
    json.dump(cui2name, f, indent=4)
