# Business Recommendation Chatbot based on Yelp Data

# How to run
## Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version

## Set up
- Download the Yelp Data at https://www.yelp.com/dataset, unzip the reviews file `yelp_dataset.tar` (4.34 GB) and put the data at `data/yelp_dataset`. For example: `data/yelp_dataset/yelp_academic_dataset_business.json`
- Create a new `.env` file based on `.env.example` and populate the variables there
- Create a new Python 3.11.9 environment: `conda create --prefix ./.venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use ./.venv`
- Install Python dependencies with Poetry: `poetry install`
- Start the Jupyterlab notebook: `poetry run jupyter lab`

# Build the RAG pipeline
- Sequentially run the notebooks 001 and 003 to prepare the data: [notebooks/001-sample-yelp-dataset](notebooks/001-sample-yelp-dataset.ipynb), [notebooks/003-collate-metadata-into-review.ipynb](notebooks/003-collate-metadata-into-review.ipynb)
- Run the notebook 005 to fine-tune the embedding model [notebooks/005-finetune-embeddings.ipynb](notebooks/005-finetune-embeddings.ipynb). On a machine with 4 GB of vRAM GPU, the fine-tuning process would take about 2 hours.
- Then explore main RAG-building notebook it [notebooks/009-pipeline-v1.ipynb](notebooks/009-pipeline-v1.ipynb) 

# Start Chatbot UI
- Navigate to `ui` folder: `cd ui`
- Run: `poetry run chainlit run chat_v2.py -hw`
- Access the Chatbot UI at http://localhost:8000
