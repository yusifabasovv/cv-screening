
# CV Screening Project

The **CV Screening** project is designed to streamline the process of evaluating job applicants' CVs against specific job requirements. Using natural language processing techniques, the project generates a summary of the CV and assesses the suitability of an applicant for a given job based on the provided requirements. Applicants submit their CVs in PDF format, while job requirements are uploaded in DOCX format.

## Folder Structure

The project has the following folder structure:

```
cv-screening-project/
│
├── main.py
├── config.yml
├── utils.py
├── requirements.txt
├── LICENSE
├── README.md
├── app.py

```

## Project Overview

The **CV Screening** project boasts the following key features:

1. **Applicant Evaluation**: The project employs OpenAI's GPT models to evaluate the compatibility of an applicant's CV with the specified job requirements.

2. **User-Friendly Interface**: The project presents a user-friendly web interface powered by the Gradio library, enabling users to seamlessly upload CVs and job requirement files, fine-tune advanced model parameters, and generate suitability scores.

3. **Exportable Results**: Users have the option to export the evaluation results to a CSV file for further analysis.

## Installation and Setup

To set up the **CV Screening** project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/cv-screening-project.git
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

3. Obtain an OpenAI API Key: In order to utilize the GPT models, you'll need an API key from OpenAI. Place your API key in the `config.yml` file.

4. Launch the Application: Execute the `main.py` script to launch the Gradio interface.

```bash
python main.py
```

5. Access the Interface: Open a web browser and navigate to the provided URL to access the **CV Screening** interface.

## Usage

1. **Upload CV**: Upload the applicant's CV in PDF format.

2. **Upload Requirements**: Upload the job requirements in DOCX format.

3. **Advanced Options**: Customize the model's behavior using advanced options like temperature and model type.

4. **Generate Output**: Click the "Generate output" button to initiate the evaluation process.

5. **View Results**: The output dataframe will display the applicant's name, suitability score, and summary.

6. **Export Results**: Click the "Export" button to export the results to a CSV file.

## Advanced Options

- **Temperature**: The temperature parameter influences the variability and creativity of the model's responses. Higher values (closer to 1) lead to greater randomness, while lower values (closer to 0) make responses more deterministic.

- **Model Type**: Choose between "gpt-3.5-turbo" and "gpt-4" for selecting the appropriate model.

## Notes

- The project's web interface leverages the Gradio library to provide a user-friendly and interactive experience for CV screening.

- Asynchronous processing is implemented using asyncio to efficiently handle multiple CV evaluations.

- The `utils.py` file contains utility functions for reading YAML configuration and formatting prompts.

## Contribution

Contributions to the **CV Screening** project are highly welcome! If you encounter any issues, have suggestions for improvements, or would like to contribute to the codebase, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---


``` !This doc is also generated by chatgpt ```
