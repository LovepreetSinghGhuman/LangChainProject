import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

path = "configs/"

# Load API key from JSON file
with open(path + "config.json", "r") as f:
    config = json.load(f)

api_key = config["groq_api_key"]

llm = ChatGroq(
    model="openai/gpt-oss-120b",
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

long_article = """LangChain: The Complete Guide to Building LLM-Powered Applications
A deep dive into the framework that changed how developers build with large language models

Introduction
When OpenAI released GPT-3 in 2020, developers immediately saw the potential. You could ask this model almost anything and get a remarkably coherent answer. But building real applications on top of it — applications with memory, tools, data, and logic — was a different story entirely. There was no standard way to chain prompts together. No clean abstraction for giving models access to external information. No pattern for making them take actions in the world. Every team was reinventing the same plumbing from scratch.
LangChain changed that. Released in October 2022 by Harrison Chase, it arrived at exactly the right moment: just weeks before ChatGPT set the internet on fire and sent every company scrambling to build AI-powered products. LangChain gave developers a shared vocabulary and a coherent set of building blocks. It became, almost instantly, one of the fastest-growing open-source projects in history — reaching 10,000 GitHub stars in the first few months and tens of thousands more as the AI wave crested.
Today LangChain is mature, opinionated, and used in production by thousands of companies. It has also evolved considerably, spawning a whole ecosystem (LangSmith, LangServe, LangGraph) that takes it well beyond its original scope. This article covers everything: what LangChain actually is, how its pieces fit together, where it shines, where it struggles, and how to use it effectively.

What LangChain Actually Is
LangChain is a framework for building applications powered by large language models. That's a broad description, and intentionally so. The framework doesn't commit you to a particular LLM, a particular vector store, or a particular deployment strategy. Instead, it provides:

Abstractions over the many different LLMs, embeddings, and vector databases in the ecosystem, so you can swap components without rewriting your application logic
Composable primitives — prompts, parsers, retrievers, chains — that can be wired together into complex pipelines
Pre-built patterns for the most common use cases: question answering over documents, conversational agents, structured data extraction, and more
Tooling for observability (LangSmith), serving (LangServe), and orchestration (LangGraph)

The core insight behind LangChain is that most LLM applications share the same structure. You need to construct a prompt, call a model, parse the output, possibly call some tools or retrieve some documents, and repeat until you have a final answer. LangChain makes those steps modular and composable.

The Architecture: From Chains to LCEL
The Original Chain Model
In its earliest versions, LangChain was organized around a concept called chains — classes that encapsulated a sequence of operations. You'd instantiate an LLMChain with a prompt and a model, or a SequentialChain that fed the output of one chain into the next. This worked, but the class hierarchy became increasingly unwieldy. Every new use case spawned a new chain class, and the codebase ballooned accordingly."""

summary = chain.invoke({"article": long_article})

print(summary)