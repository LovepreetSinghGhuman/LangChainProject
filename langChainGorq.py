import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load API key from JSON file
with open("groq_config.json", "r") as f:
    config = json.load(f)
api_key = config["groq_api_key"]

llm = ChatGroq(
    model="llama-3.3-70b-versitile",
    temperature=0.7,
    api_key=api_key
)

template = """
Summarize the following article in 1 concise sentence:

{article}
    """
    
prompt = ChatPromptTemplate.from_template(template)

parser = StrOutputParser()

chain = prompt | llm | parser

long_article = """The article discusses the recent advancements in artificial intelligence, 
particularly focusing on the development of large language models. 
It highlights the capabilities of these models in understanding and generating human-like text,
as well as their applications in various industries such as healthcare, finance, and customer service. The article also addresses the ethical considerations surrounding the use of AI, 
including concerns about bias, privacy, and the potential for job displacement. 
It emphasizes the importance of responsible AI development and the need for regulations to ensure that these technologies are used for the benefit of society. 
Overall, the article provides an overview of the current state of AI and its implications for the future."""

summary = chain.invoke({"article": long_article})

print(summary)