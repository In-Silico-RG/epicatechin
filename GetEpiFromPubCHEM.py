import requests

cid = 72276  # Epicatechin PubChem CID
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    record = data['PC_Compounds'][0]

    for prop in record['props']:
        label = prop['urn']['label']
        value = prop['value']
        if label == 'Molecular Weight':
            molecular_weight = value.get('fval') or value.get('sval') or value.get('ival')
            print("Molecular Weight:", molecular_weight)
        elif label == 'IUPAC Name':
            iupac_name = value.get('sval')
            print("IUPAC Name:", iupac_name)
        elif label == 'SMILES' and prop['urn'].get('name') == 'Canonical':
            smiles = value.get('sval')
            print("Canonical SMILES:", smiles)
else:
    print("Failed to retrieve data:", response.status_code, response.text)

#The script loops through each property (prop) in the compound record.
It checks the label (e.g., 'Molecular Weight', 'IUPAC Name', 'SMILES') 
and prints the corresponding value.

