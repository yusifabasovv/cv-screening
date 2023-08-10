import docx2txt
import json
import pdf2docx

import openai
import gradio as gr

import utils

files = utils.read_yaml('config.yml')

API_KEY = files["API_KEY"]
PROMPT = files["PROMPT"]

openai.api_key = API_KEY



class cv_screening:
    """Class for preprocessing files. """

    @classmethod
    def convert_pdf_to_docx(cls, pdf_file_path, docx_file_path):
        try:
            pdf2docx.parse(pdf_file_path, docx_file_path)
        except Exception as e:
            print(f"Error: {e}")

    @classmethod
    def extract_text_from_docx(cls, docx_file_path):
        try:
            docx_file = docx_file_path.name  # if uploaded as file via gradio
        except:
            docx_file = docx_file_path

        return docx2txt.process(docx_file)

    @classmethod
    def get_completion(cls, prompt: str, model: str, temperature: int=0.7):

        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
        return response.choices[0].message["content"]



def validate_candidate(pdf_file_path, docx_requirements_path, temperature, model: str = "gpt-3.5-turbo"):
    """ Runs the gpt model with given prompt 
    
    
    Inputs: 
      pdf_file_path - path to cv file.
      docx_requirements_path - path to requirements file for position.
      temperature - gpt model parameter.
      model - gpt model type

    Output:
        Output text my model
    """

    try:
        pdf_file = pdf_file_path.name # if uploaded as file via gradio
    
    except:
        pdf_file = pdf_file_path
    
    docx_file_path = pdf_file.split('.pdf')[0]+'.docx'
    
    cv_screening.convert_pdf_to_docx(pdf_file, docx_file_path)
    pdf_text = cv_screening.extract_text_from_docx(docx_file_path)
    docx_text = cv_screening.extract_text_from_docx(docx_requirements_path)
    
    prompt = str(PROMPT.format(docx_text, pdf_text))

    return cv_screening.get_completion(prompt, temperature=temperature, model=model)


def run_gradio():
    # Create the Gradio interface
    iface = gr.Interface(
        fn=validate_candidate,
        inputs=[
            gr.File(label="Upload PDF Resume", type="file", file_types=['.pdf']),
            gr.File(label="Upload Requirements text", type="file", file_types=['.docx']),
            gr.Slider(minimum=0, maximum=1, value=0.5,
                    label="Model temperature", 
                    info="Temperature is a parameter of the OpenAI (ChatGPT, GPT-3, and GPT-4) models that influence the variability and consequently the creativity of responses"),
            gr.Dropdown(choices=["gpt-3.5-turbo", "gpt-4"], value="gpt-3.5-turbo", label="Model type")


        ],
        outputs=gr.Textbox(label="Resume summary", show_copy_button=True,
                        show_label=True, lines=10),
        title="CV Screening",
        description="Upload a PDF resume and a text file, get summary text.",
    )

    # Launch the interface
    iface.launch(share=True)




if __name__ == "__main__":
    #run_gradio()
    print(validate_candidate('files/Lalə Qaralı Nadirova.pdf', 
                    'files/Advanced Analytics Product Owner EN - 06072023.docx',
                   temperature=0.7))
