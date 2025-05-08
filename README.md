# AI Resume Optimizer

This AI workflow uses HuggingFace-hosted LLMs (e.g. Qwen, Mixtral) to optimize your resume for a specific job description. It rewrites your resume to emphasize relevant skills, experiences, and keywords.
It uses [LangGraph](https://github.com/langchain-ai/langgraph) to define a simple stateful workflow and [LangChain’s HuggingFace integration](https://python.langchain.com/docs/integrations/llms/huggingface_hub/) to run the model.

## 🧠 How It Works

1. Paste your resume and a target job description.
2. The model rewrites the resume using a carefully designed prompt.
3. You receive a tailored version ready to use in job applications.

## 🚀 Features

- Reads a resume and job description from local files
- Uses a HuggingFace model (e.g. `Qwen/Qwen2.5-72B-Instruct`) to generate a tailored resume
- Highlights relevant experience, skills, and technologies
- Outputs a rewritten resume as `new_resume.txt`

## 🛠️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/galzmarc/tiny_agent
   cd tiny_agent

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3. In the same folder, create a resume.txt and a description.txt files.

4. Set your HuggingFace token in a .env file
    ```bash
    HF_TOKEN="your_huggingface_token_here"
    ```

5. Run the script:
    ```bash
    python3 main.py
    ```
6. A new file new_resume.txt will be created with the optimized resume.