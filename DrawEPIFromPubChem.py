import requests
from rdkit import Chem
from rdkit.Chem import Draw

# Step 1: Get epicatechin data from PubChem
cid = 72276  # Epicatechin CID
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/ConnectivitySMILES/JSON"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    try:
        smiles = data['PropertyTable']['Properties'][0]['ConnectivitySMILES']
        print("Epicatechin SMILES:", smiles)
    except (KeyError, IndexError) as e:
        print("Could not find ConnectivitySMILES in the response:", e)
        smiles = None
else:
    print("Failed to get data from PubChem:", response.status_code, response.text)
    smiles = None

# Step 2: Draw 2D image using RDKit
if smiles:
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(300, 300))
    img.show()  # Display the image in Jupyter or compatible environments
    img.save("epicatechin_2d.png")  # Save the image to a file

# To draw a 2D image of epicatechin after retrieving its information from a free 
database like PubChem, you can use the RDKit library to visualize its molecular 
structure. RDKit can generate 2D images from SMILES strings or other chemical identifiers.

