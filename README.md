# text-classification

A repository for training and testing multiple text classification models.

## Requirements

- Python

## Setup

1. Create new virtual environment

        python -m venv .venv

2. Activate virtual environment

    Windows:

        . ./.venv/Scripts/activate

    Linux:

        . ./.venv/bin/activate

3. Install requirements

        pip install nltk xgboost sklearn datasets transformers evaluate gensim numpy Sastrawi torch

## Usage

To see available commands, execute this command:

    python main.py -h

Example command: start training on BERT model:

    python main.py train bert
    