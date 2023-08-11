import asyncio
import time
import json
import gradio as gr
import pandas as pd
import pdf2docx
import docx2txt
import openai
from typing import List, Union
import utils

# Read configuration from YAML file
files = utils.read_yaml("config.yml")
API_KEY = files["API_KEY"]
PROMPT = files["PROMPT"]

openai.api_key = API_KEY

class CVScreening:
    """Class for preprocessing files."""

    @classmethod
    def convert_pdf_to_docx(cls, *, pdf_file_path: str, docx_file_path: str) -> None:
        """
        Convert a PDF file to DOCX format.
        
        Args:
            pdf_file_path (str): Path to the input PDF file.
            docx_file_path (str): Path to save the converted DOCX file.
        """
        try:
            pdf2docx.parse(pdf_file_path, docx_file_path)
        except Exception as e:
            print(f"Error: {e}")

    @classmethod
    def extract_text_from_docx(cls, *, docx_file_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            docx_file_path (str): Path to the input DOCX file.
            
        Returns:
            str: Extracted text from the DOCX file.
        """
        try:
            docx_file = docx_file_path.name if hasattr(docx_file_path, 'name') else docx_file_path
            return docx2txt.process(docx_file)
        except Exception as e:
            print(f"Error: {e}")
            return ""

    @classmethod
    async def get_completion(
        cls,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.7
    ) -> str:
        """
        Get model completion using OpenAI's Chat API.
        
        Args:
            prompt (str): Input prompt for the conversation.
            model (str): GPT model type.
            temperature (float): GPT model temperature parameter.
            
        Returns:
            str: Generated text by the model.
        """
        messages = [{"role": "user", "content": prompt}]
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        
        time.sleep(5)

        return response.choices[0].message["content"]

async def validate_candidate(
    pdf_file_paths: List[str],
    docx_requirements_path: str,
    temperature: float,
    model: str = "gpt-3.5-turbo"
) -> pd.DataFrame:
    """
    Perform candidate validation using GPT model.
    
    Args:
        pdf_file_paths (List[str]): Paths to CV files in PDF format.
        docx_requirements_path (str): Path to requirements file for the position in DOCX format.
        temperature (float): GPT model temperature parameter.
        model (str): GPT model type.
        
    Returns:
        pd.DataFrame: Generated text by the model.
    """
    
    docx_text = CVScreening.extract_text_from_docx(docx_file_path=docx_requirements_path)
    
    prompts = []
    
    for pdf in pdf_file_paths:
        
        try:
            pdf_file = pdf.name if hasattr(pdf, 'name') else pdf
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()

        docx_file_path = pdf_file.split(".pdf")[0] + ".docx"

        CVScreening.convert_pdf_to_docx(pdf_file_path=pdf_file, docx_file_path=docx_file_path)
        pdf_text = CVScreening.extract_text_from_docx(docx_file_path=docx_file_path)
        
        prompt = str(PROMPT.format(docx_text, pdf_text))
        prompts.append(prompt)
           
    tasks = [CVScreening.get_completion(prompt=i, temperature=temperature, model=model) for i in prompts]
    
    L = await asyncio.gather(*tasks)
    return pd.concat([pd.DataFrame(json.loads(i)).T for i in L]).reset_index().rename({"index": "name"}, axis=1)

def run_gradio() -> None:
    # Create the Gradio interface
    iface = gr.Interface(
        fn=validate_candidate,
        inputs=[
            gr.File(label="Upload PDF Resume", type="file", file_types=[".pdf"], file_count="multiple"),
            gr.File(label="Upload Requirements text", type="file", file_types=[".docx"]),
            gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                label="Model temperature",
                info="Temperature is a parameter of the OpenAI (ChatGPT, GPT-3, and GPT-4) models that influence the variability and consequently the creativity of responses",
            ),
            gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-4"],
                value="gpt-3.5-turbo",
                label="Model type",
            ),
        ],
        outputs=gr.DataFrame(
            datatype=["str", "number", "str"],
            headers=["Name", "Score", "Summary"]
        ),
        title="CV Screening",
        description="Upload a PDF resume and a text file, get summary text.",
    )

    # Launch the interface
    iface.launch()


if __name__ == "__main__":
    run_gradio()

