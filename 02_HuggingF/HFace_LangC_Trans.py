
# =============================
# Imports
# =============================

from langchain.prompts import PromptTemplate
from transformers import pipeline, logging as hf_logging
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch

# =============================
# Silence Transformers Warnings & Progress Bars
# =============================
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


# =============================
# Utility Functions
# =============================
def print_torch_info():
    print("PyTorch version:", torch.__version__)
    print("CUDA/Rocm available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")


# =============================
# Main Summarization Logic
# =============================
def main():
    print_torch_info()

    # --- Model pipelines ---

    model_fb = pipeline(
        task="summarization", 
        model="facebook/bart-large-cnn",
        progress_bar=False
    )


    model_google = pipeline(
        task="summarization",
        model="google/pegasus-xsum",  # A robust summarization model
        tokenizer="google/pegasus-xsum",
        use_fast=False,
        progress_bar=False
    )

    # --- Example summarization ---
    example_article = """
    LangChain: The Complete Guide to Building LLM-Powered Applications
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
    In its earliest versions, LangChain was organized around a concept called chains — classes that encapsulated a sequence of operations. You'd instantiate an LLMChain with a prompt and a model, or a SequentialChain that fed the output of one chain into the next. This worked, but the class hierarchy became increasingly unwieldy. Every new use case spawned a new chain class, and the codebase ballooned accordingly.
    """

    print("\n--- Bart-Large-CNN Summary ---")
    print(model_fb(example_article))

    # --- LangChain pipeline with Google Pegasus ---
    llm = HuggingFacePipeline(pipeline=model_google)
    template = PromptTemplate.from_template(
        "Summarize the following article in 1 concise sentence:\n\n{article}\n\nSummary:"
    )
    chain = template | llm

    topic = input("\nEnter the article to summarize: ")
    response_google = chain.invoke({"article": topic})
    print("Google Summary:", response_google)


if __name__ == "__main__":
    main()