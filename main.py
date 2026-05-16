# SPDX-License-Identifier: 0BSD OR Apache-2.0
# Copyright (C) 2025 Quilldrake Labs
# Copyright (C) 2026 Madeleine Choi

import unicodedata
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv


def normalise_tag(tag: str) -> str:
    tag = tag.lower()
    tag = unicodedata.normalize("NFKC", tag)
    tag = tag.strip()

    return tag


def main():
    load_dotenv()

    danbooru_path = Path("data/danbooru.pkl")
    danbooru: pd.Series

    if not danbooru_path.exists():
        dataset = load_dataset("qdlabs/danbooru-tags")
        data = pd.DataFrame(dataset["train"])

        data = data.query("category == 0")  # general
        data = data.query("post_count >= 10")

        data.dropna(subset=["name"], inplace=True)

        # this nukes a lot we don't want: emoticons, tank models, etc.
        data = data[data["name"].str.contains(r"^[a-z][a-z ]*$", regex=True)]

        data["name"] = data["name"].apply(normalise_tag)  # type: ignore

        danbooru = cast(pd.Series, data["name"])
        danbooru.to_pickle(danbooru_path)
    else:
        danbooru = cast(pd.Series, pd.read_pickle(danbooru_path))

    wiktionary_path = Path("data/wiktionary.pkl")
    wiktionary: pd.Series

    if not wiktionary_path.exists():
        # data/wiktionary.txt is just a newline-separated list of en lemmas and
        # non-lemma forms, use petscan to grab it
        txt = Path("data/wiktionary.txt")
        assert txt.exists()

        with open(txt, encoding="utf-8") as f:
            wiktionary = pd.Series(line.rstrip("\n").lower() for line in f)

        wiktionary = cast(
            pd.Series, wiktionary[~wiktionary.str.contains(r"^\d| ", regex=True)]
        )
        wiktionary.to_pickle(wiktionary_path)
    else:
        wiktionary = cast(pd.Series, pd.read_pickle(wiktionary_path))

    missing = cast(pd.Series, danbooru[~danbooru.isin(wiktionary)])
    missing.to_csv("data/missing.txt", sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()
