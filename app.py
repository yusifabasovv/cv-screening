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
    def convert_pdf_to_docx(cls, pdf_file_path, docx_file_path):
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
    def extract_text_from_docx(cls, docx_file_path):
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
    def export_csv(cls, *, output: Union[pd.DataFrame, pd.Series]) -> gr.File:
        """
        Export data to a CSV file.
        
        Args:
            output (Union[pd.DataFrame, pd.Series]): Data to be exported.
            
        Returns:
            gr.File: A Gradio File object representing the exported CSV file.
        """
        file_name = "applicants.csv"
        output.to_csv(file_name)
        return gr.File.update(value=file_name, visible=True)
    
    
    
    @classmethod
    async def get_completion(cls, prompt: str, model: str, temperature: float = 0.7):
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
    

async def validate_candidate(pdf_file_paths, docx_requirements_path, temperature: float=0.7, model: str = "gpt-3.5-turbo"):
    """
    Perform candidate validation using GPT model.
    
    Args:
        pdf_file_path (str): Path to CV file in PDF format.
        docx_requirements_path (str): Path to requirements file for the position in DOCX format.
        temperature (float): GPT model temperature parameter.
        model (str): GPT model type.
        
    Returns:
        str: Generated text by the model.
    """
    
    docx_text = CVScreening.extract_text_from_docx(docx_requirements_path)
    
    prompts = []
    
    for pdf in pdf_file_paths:
        
        try:
            pdf_file = pdf.name if hasattr(pdf, 'name') else pdf
        except Exception as e:
            print(f"Error: {e}")
            return ""

        docx_file_path = pdf_file.split(".pdf")[0] + ".docx"

        CVScreening.convert_pdf_to_docx(pdf_file, docx_file_path)
        pdf_text = CVScreening.extract_text_from_docx(docx_file_path)
        
        prompt = str(PROMPT.format(docx_text, pdf_text))
        prompts.append(prompt)
           
    tasks = [get_completion(i, temperature=temperature, model=model) for i in prompts]
    
    L = await asyncio.gather(*tasks)
    return pd.concat([pd.DataFrame(json.loads(i)).T for i in L]).reset_index().rename({"index": "name"}, axis=1)



    
def run_gradio():
    with gr.Blocks() as demo:
        
        gr.Markdown('# CV Screening')
        with gr.Tab("Input "):
            pdf_output = gr.File(label="Upload PDF Resume", 
                                type="file", file_types=[".pdf"], 
                                 file_count = "multiple")

            docx_output = gr.File(label="Upload Requirements text", type="file", file_types=[".docx"])

            with gr.Accordion(label="Advanced options", open=False):

                temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.05,
                        label="Model temperature",
                        info="Temperature is a parameter of the OpenAI (ChatGPT, GPT-3, and GPT-4) models that influence the variability and consequently the creativity of responses",
                    ),
                model = gr.Dropdown(
                        choices=["gpt-3.5-turbo", "gpt-4"],
                        value="gpt-3.5-turbo",
                        label="Model type",
                        interactive=True
                    ),

            btn = gr.Button("Generate output")
            clear = gr.ClearButton(components=[pdf_output, docx_output], value="Clear console")

        with gr.Tab("Output dataframe"):

            output = gr.DataFrame(datatype=["str", "number", "str"],
                                      headers = ["Name", "Score", "Summary"])


            with gr.Column():
                button = gr.Button("Export")
                csv = gr.File(interactive=False, visible=False)

            button.click(CVScreening.export_csv, output, csv)



        btn.click(validate_candidate, inputs=[pdf_output, docx_output, list(temperature)[0], 
                                              list(model)[0]], outputs=output)
    demo.queue().launch()
    

if __name__ == "__main__":
    run_gradio()

