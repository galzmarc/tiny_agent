import os
from typing import TypedDict
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END


### Setup

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    #repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",    # access needs to be requested
    task='text-generation',
    temperature=0.5,
    max_new_tokens=1024,
    huggingfacehub_api_token=hf_token
)

### Tools

@tool
def read_file(path: str) -> str:
    """
    Reads the contents of a file in the provided path.
    
    Args:
        path: the relative path of a file in the working directory.
    
    Returns:
        The content of the file
    """
    with open(path, "r") as f:
        return f.read()
    
@tool
def write_file(path: str, content: str) -> str:
    """
    Writes the provided content in a new file, and returns the name of the newly created file.
    
    Args:
        path: the relative path of a new file in the working directory.
        content: the content to write in the new file.
    
    Returns:
        The name of the newly created file
    """
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"New file created: {path}"
    except FileExistsError:
        return f"File already exists: {path}"
    except Exception as e:
        return f"Error writing file: {e}"

### State

class ResumeState(TypedDict):
    resume_data: str
    job_data: str
    tailored_resume: str

def read_resume(state):
    state["resume_data"] = read_file.invoke({"path": "resume.txt"})
    return state

def read_description(state):
    state["job_data"] = read_file.invoke({"path": "description.txt"})
    return state

def tailor_resume(state):
    prompt = f"""
    You are a resume optimization assistant.
    Your goal is to improve and tailor the resume to match the job description.
    Identify keywords and required skills, and emphasize matching qualifications.

    Guidelines:
    - Do NOT delete any sections (e.g., Education, Projects, Certifications, etc.).
    - Do not modify any job titles, companies, or dates in the WORK EXPERIENCE section.
    - Rewrite bullet points to emphasize alignment with the job description.
    - Integrate relevant keywords and responsibilities from the job description.
    - Select only the skills and projects that are most relevant to the job description.
    - Do not include task lists or explanation — return only the modified resume.

    Here is the original resume:\n
    {state['resume_data']}
    \n\n
    Here is the job description:\n
    {state['job_data']}
    """

    res = llm.invoke(prompt)
    state["tailored_resume"] = res.strip() if isinstance(res, str) else str(res)
    return state

def save_resume(state):
    write_file.invoke({"path": "new_resume.txt", "content": state["tailored_resume"]})
    return state


### Main

def main():

    # Build graph
    builder = StateGraph(ResumeState)
    builder.add_node("read_resume", read_resume)
    builder.add_node("read_description", read_description)
    builder.add_node("tailor_resume", tailor_resume)
    builder.add_node("save_resume", save_resume)
    
    # Logic
    builder.add_edge(START, "read_resume")
    builder.add_edge("read_resume", "read_description")
    builder.add_edge("read_description", "tailor_resume")
    builder.add_edge("tailor_resume", "save_resume")
    builder.add_edge("save_resume", END)

    # Add
    graph = builder.compile()
    graph.invoke({})

if __name__ == "__main__":
    main()