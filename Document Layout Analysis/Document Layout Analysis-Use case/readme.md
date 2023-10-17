**Document Layout Analysis Project**

**Introduction**

Welcome to the Document Layout Analysis project! This project allows you to perform document layout analysis on images or PDFs to extract information about their structure and content. This README.md file will guide you through the setup and usage of both the UI and API components of the project.

**Prerequisites**

Before you get started, make sure you have the following prerequisites installed on your system:

Python 3.8 or higher
virtualenv (optional but recommended)

**Installation**

1. Clone this repository to your local machine.

2. Change your working directory to the project folder.

3. Create a virtual environment (optional but recommended):
   
Code: python -m venv venv
   
5. Activate the virtual environment:
   
On Windows:
code: venv\Scripts\activate

On macOS and Linux:
code: source venv/bin/activate

6. Install project dependencies using the provided requirements files:

code:
pip install -r detectronreq.txt
pip install -r layoutparserreq.txt
pip install -r requirements.txt

**Usage**

Once you have successfully installed the project and its dependencies, you can start using the Document Layout Analysis tool.

**User Interface (UI)**
To use the User Interface:

1. Ensure your virtual environment is activated (if you created one in the installation step).
   
2. Run the UI Flask application:

Code:
python DocumentLayoutAnalysisUI.py

4. Open your web browser and navigate to http://localhost:80 to access the user interface.
5. Upload the document (image or PDF) you want to analyze.
6. Click the "Analyze" button to initiate the layout analysis.
7. The tool will process the document and provide you with information about its structure and content.

**API**
To use the API:

1. Ensure your virtual environment is activated (if you created one in the installation step).
   
2. Run the API Flask application:
   
code:
python DocumentLayoutAnalysisAPI.py

3. The API is now running and can accept requests at http://localhost:5000/api.
   
4. You can send a POST request to the /api/layout-analyzer endpoint with the document (image or PDF) you want to analyze.
   
5. The API will process the document and return the analysis results in JSON format.

**Running as a Docker Container**
To run the UI Flask application as a Docker container:

1. Make sure you have Docker installed on your system.
   
2. Build the Docker image from the project directory:
   
Code:
docker build -t document-layout-analysis-ui .

3. Run the Docker container:

Code:
docker run -p 8000:80 document-layout-analysis-ui
Open your web browser and navigate to http://localhost:8000 to access the user interface running inside the Docker container.

Happy Document Layout Analysis!
