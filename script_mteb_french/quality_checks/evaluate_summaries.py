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
                model="gpt-3.5-turbo-0125", 
                temperature=0, api_key=os.getenv("OPENAI_API_KEY"), 
                openai_organization=os.getenv("OPENAI_ORGANIZATION"),
            )

    print("Load datasets from HF..")
    data_fr = load_dataset("lyon-nlp/summarization-summeval-fr-p2p", split="test")
    data_en = load_dataset("mteb/summeval", split="test")
    print(data_fr, data_en)

    prompt = """Given an original text in English and its translation in French, \
        evaluate the quality of the translation only with rates between 1 and 5 with 1 being not of good quality,\
        and 5 of being very good.\n No need to mind the quality of the text as original English text may be of bad quality."""


    print("\nRun evaluations..\n")
    for fr_texts in data_fr["machine_summaries"]:
        for eng_texts in data_en["machine_summaries"]:
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
        break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_type", type=str, default="azure", choices=["azure", "openai"])
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)