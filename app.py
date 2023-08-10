import docx2txt
import pdf2docx
import openai
import gradio as gr
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
    def get_completion(cls, prompt: str, model: str, temperature: float = 0.7):
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
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message["content"]

def validate_candidate(pdf_file_path, docx_requirements_path, temperature, model: str = "gpt-3.5-turbo"):
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
    try:
        pdf_file = pdf_file_path.name if hasattr(pdf_file_path, 'name') else pdf_file_path
    except Exception as e:
        print(f"Error: {e}")
        return ""

    docx_file_path = pdf_file.split(".pdf")[0] + ".docx"

    CVScreening.convert_pdf_to_docx(pdf_file, docx_file_path)
    pdf_text = CVScreening.extract_text_from_docx(docx_file_path)
    docx_text = CVScreening.extract_text_from_docx(docx_requirements_path)

    prompt = str(PROMPT.format(docx_text, pdf_text))

    return CVScreening.get_completion(prompt, temperature=temperature, model=model)

def run_gradio():
    # Create the Gradio interface
    iface = gr.Interface(
        fn=validate_candidate,
        inputs=[
            gr.File(label="Upload PDF Resume", type="file", file_types=[".pdf"]),
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
        outputs=gr.Textbox(
            label="Resume summary", show_copy_button=True, show_label=True, lines=10
        ),
        title="CV Screening",
        description="Upload a PDF resume and a text file, get summary text.",
    )

    # Launch the interface
    iface.launch(share=True)

if __name__ == "__main__":
    run_gradio()

