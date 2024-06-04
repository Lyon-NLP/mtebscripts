import os

from dotenv import load_dotenv

from datasets import load_dataset
from tqdm import tqdm
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from sklearn.metrics import f1_score, classification_report

DATASET = "lyon-nlp/clustering-hal-s2s"
SEED = 42

load_dotenv()


dataset = load_dataset(DATASET, name="mteb_eval", split="test")
dataset = dataset.class_encode_column("domain")
id2label = {k: dataset.features["domain"].int2str(k) for k in range(dataset.features["domain"].num_classes)}
dataset = dataset.train_test_split(test_size=0.3, shuffle=True, stratify_by_column="domain", seed=SEED)

X_test, y_test = dataset["test"]["title"], dataset["test"]["domain"]

model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)

prompt = """\
Given a title of a scientific paper, classify it into the right domain from the list below.
Given domains and their descriptions:
- shs: Sciences humaines et sociales
- sdv: Sciences du vivant (Biologie)
- spi: Sciences de l'ingénieur (Physique)
- info: Informatique
- sde: Sciences de l'environnement
- phys: Physique
- sdu: Planète et Univers (Physique)
- math: Mathématiques
- chim: Chimie
- scco: Sciences cognitives
- qfin: Économie et finance quantitative
- stat: Statistiques
- other: Autre
- stic:
- nlin: Science non linéaire (Physique)
- electromag: Electro-magnétisme
- instrum: Instrumentation (Physique)
- image: Image

Provide the classification in the following format:
Classification:::
Domain: (domain_name: e.g., shs, sdu, math, etc.)

Examples:
Title: "La transformation digitale du management des ressources humaines et de ses enjeux pour les entreprises"
Domain: shs

Title: "Sur l'approximation numérique de quelques problèmes en mécanique des fluides"
Domain: math

Now here is the title of the paper:
Title: {title}

Classification:::
Domain: 
"""

preds, targets = [], []
for i, sample, label in tqdm(zip(range(len(X_test)), X_test, y_test)):
    message = HumanMessage(
        content=prompt.format(title=sample),
        temperature=0,
    )
    response = model.invoke([message])  
    pred = response.content.split(":")[1].strip() if ":" in response.content else response.content
    preds.append(pred)
    targets.append(id2label[label])

    if i % 50 == 0:
        print(f"Processed {i} samples")
        print(preds, targets)
        f1 = f1_score(targets, preds, average="macro")
        print(f1)
    if i == 599:
        break
    
f1 = f1_score(targets, preds, average="macro")
print(f1)
    