"""Evaluate the quality of translation between English and French texts using OpenAI API.

* Set the following environment variables before running the script with api_type="azure":

```bash
export AZURE_OPENAI_API_KEY="API_KEY"
export AZURE_OPENAI_ENDPOINT="https://<ENDPOINT_NAME>.openai.azure.com"
export AZURE_OPENAI_API_VERSION="API_VERSION"
export AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="DEPLOYMENT_NAME"
```	

* Set the following environment variables before running the script with api_type="openai":

```bash
export OPEN_API_KEY="API_KEY"
export OPENAI_ORGANIZATION="ORGANIZATION_NAME"
```
"""

import os
import argparse

from datasets import load_dataset
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI


def main(args):
    match args.api_type:   
        case "azure":
            model = AzureChatOpenAI(
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            )
        case "openai":
            model = ChatOpenAI(
                model=args.model_name, 
                temperature=0, api_key=os.getenv("OPENAI_API_KEY"), 
                openai_organization=os.getenv("OPENAI_ORGANIZATION"),
            )

    data_fr = load_dataset("lyon-nlp/summarization-summeval-fr-p2p", split="test")
    data_en = load_dataset("mteb/summeval", split="test")
    print(data_fr, data_en)

    prompt = """Given an original text in English and its translation in French, \
        evaluate the quality of the translation only with rates {"Excellent": 4, "Acceptable": 3, "Could be Improved": 2, "Bad": 1}.\
        No need to mind the quality of the text as original English text may be of bad quality."""

    for fr_texts, eng_texts in zip(data_fr["machine_summaries"], data_en["machine_summaries"]):
        for i in range(len(fr_texts)):
            content = f"Original text in English: {eng_texts[i]}\nTranslation in French: {fr_texts[i]}\n"
            message = HumanMessage(
                content=prompt+content,
                temperature=args.temperature,
            )
            response = model.invoke([message])
            print(f"\nOriginal text in English: {eng_texts[i]}")
            print(f"\nTranslation in French: {fr_texts[i]}\n")
            print(response.content)
                
            break
        break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", type=str, default="azure", choices=["azure", "openai"])
    parser.add_argument("--model_name", type=str, default="gpt4-turbo")
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)