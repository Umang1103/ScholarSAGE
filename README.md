# ScholarSAGE: LLM-powered guide to research queries

## Why ScholarSAGE?
- Understanding and navigating through the extensive body of work authored by a researcher poses a significant challenge for individuals interested in their contributions.
- Accessing relevant information efficiently is often hindered by the sheer volume of research paper.
- There exists a need for an intelligent solution that streamlines this process, offering personalized assistance and insights.
- Current methods lack the adaptability and comprehensiveness required to address the unique nature of scholarly inquiries, necessitating the development of a specialized chatbot to bridge this gap.

## Objectives
- To develop a sophisticated chatbot, by fine-tuning pre-trained Large Language Models (LLM) on the extensive research papers authored by a specific researcher. 
- Improve on performance metrics in terms of cosine similarity.

## Getting started
- Clone the repository. Create a virtual env using the command `python -m venv <path_to_new_virtual_env>`.
- Install the requirements using the command `pip install -r requirements.txt`.
- Replace the `HUGGINGFACEHUB_API_TOKEN` and `PINECONE_API_KEY` in `initializations.py` with your respective keys.
- To run the application run `streamlit run chat.py`.
