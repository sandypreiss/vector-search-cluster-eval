import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import polars as pl
import tiktoken
from jinja2 import Template
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def get_n_to_sample(
    model: str, token_target: int, df: pl.DataFrame, text_col: str
) -> int:
    """Selects the number of documents to sample from each cluster,
    such that the total number of tokens in the sample is close to
    the target number of tokens.

    Args:
        model (str): OpenAI model string to use for tokenization
        token_target (int): target total number of tokens in sample
        df (pl.DataFrame): source data
        text_col (str): column in source dataframe with texts

    Returns:
        int: number to sample from each cluster
    """
    encoding = tiktoken.encoding_for_model(model)
    mean_tokens_per_doc = pl.Series(
        [len(encoding.encode(doc)) for doc in df[text_col]]
    ).mean()
    return int(token_target / mean_tokens_per_doc)


def make_prompt_template() -> Template:
    template = """Please identify the common topic of the following news articles:

    Begin list of news articles:

    {% for doc in sample %}
    {{ doc }}
    {% endfor %}

    End list of news articles.

    Return only a concise phrase, anywhere from a single word to a few words,
    describing the common topic of the articles. 
    Don't use filler words like "various topics in...", "diverse...", etc.
    Don't include preludes like "The common topic is..."
    Avoid being overly specific. Don't include topics unless the large majority
    of the articles are about that topic. 
    Avoid compound names that consist of two topics joined by "and" - 
    strive for a name that includes both of those topics instead. 
    """
    return Template(template)


def sample_docs(docs: pl.Series, n: int) -> pl.Series:
    if len(docs) <= n:
        sampled_docs = docs
    else:
        sampled_docs = docs.sample(n)
    return sampled_docs.to_list()


@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def complete_chat_with_retry(
    client: OpenAI, model: str, system_prompt: str, user_prompt: str
) -> str:
    """
    Wrapper for OpenAI chat response implementing retry with exponential
    backoff using tenacity
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":

    MODEL = "gpt-4o-2024-05-13"
    # Staying well under this model's 128k context window
    TOKEN_TARGET = 100_000
    SYSTEM_PROMPT = """You are an expert journalism librarian.
    Your job is to review groups of news articles and identify their common topic.
    """

    load_dotenv()
    df = pl.read_parquet("data/bbc_clustered.parquet")

    n_to_sample = get_n_to_sample(
        model=MODEL,
        token_target=TOKEN_TARGET,
        df=df,
        text_col="text",
    )

    prompt_template = make_prompt_template()

    cluster_col_prefix = "cluster_from_"
    cluster_cols = [col for col in df.columns if col.startswith(cluster_col_prefix)]

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    all_names = {}
    for cluster_col in cluster_cols:
        print(f"Generating names for {cluster_col}")
        cluster_names = {}
        clusters = df[cluster_col].unique()
        for cluster in clusters:
            docs = df.filter(df[cluster_col] == cluster)["text"]
            sample = sample_docs(docs, n_to_sample)
            prompt = prompt_template.render(sample=sample)
            name = complete_chat_with_retry(
                client=client,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            cluster_names[cluster] = name
        all_names[cluster_col.replace(cluster_col_prefix, "")] = cluster_names

    # save names to json
    with open("data/cluster_names.json", "w") as f:
        json.dump(all_names, f)
