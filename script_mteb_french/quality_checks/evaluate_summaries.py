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
import json
import re

from datasets import load_dataset
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import openai


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

    prompt = """
You will be given a couple of texts in English and its translation in French.
Your task is to provide a 'rating' score on how well the system translated the English text into French.
Give your answer as a float on a scale of 0 to 10, where 0 means that the system_translation is bad and does not represent what is being said in the original English text, and 10 means that the translation is good and represents the original English text.
No need to mind the quality of the text as original English text may be of bad quality.

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a float between 0 and 10)

Now here are the English and French texts.

Original text in English: {english_text}
Translation in French: {french_translation}

Feedback:::
Total rating: """
    
    translations_rating = []

    for fr_texts, eng_texts in zip(data_fr[args.type], data_en[args.type]):
        sub_ratings = []
        print(len(fr_texts), len(eng_texts))
        assert len(fr_texts) == len(eng_texts)
        for i in range(len(fr_texts)):
            try:
                message = HumanMessage(
                    content=prompt.format(english_text=eng_texts[i], french_translation=fr_texts[i]),
                    temperature=args.temperature,
                )
                response = model.invoke([message])
                # print(f"\nOriginal text in English: {eng_texts[i]}")
                # print(f"\nTranslation in French: {fr_texts[i]}\n")
                # print(response.content)   
                digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", response.content)]
                rate = float(digit_groups[0])
                print(rate)
                sub_ratings.append(rate)
            except openai.BadRequestError:
                print("The response was filtered due to the prompt triggering Azure OpenAI's content management policy.\nRating set to None.")
                sub_ratings.append(None)
        translations_rating.append(sub_ratings)

        # Save at each step because sometimes of openai.BadRequestError
        with open(f"translation_ratings_{args.type}.json", "w") as fp:
            json.dump(translations_rating, fp)
            


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", type=str, default="azure", choices=["azure", "openai"])
    parser.add_argument("--model_name", type=str, default="gpt4-turbo")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--type", type=str, default="human_summaries", choices=["machine_summaries", "human_summaries"])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)