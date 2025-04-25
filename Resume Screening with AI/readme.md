# Resume Matcher with Google Gemini, Langchain, Langgraph, Langsmith and FastAPI

This project uses **FastAPI** and **Google Gemini** to build an AI-powered **Resume Matcher** web application. It helps automate the resume screening process by analyzing uploaded resumes and comparing them with a given job description. The results are returned with a match score and detailed analysis in a **clean, readable format**.

## 🚀 Features
- Upload **PDF** or **DOCX** resumes and a **Job Description**.
- **Google Gemini** analyzes the resumes and compares them with the job description.
- Results are displayed with a **match score** and a breakdown of relevant skills, experience, and education.
- Uses **Markdown** for structured, clean output, which is then rendered as HTML.
- User-friendly **FastAPI** backend with **Bootstrap** frontend.

## 🛠️ Technologies Used
- **FastAPI**: For building a fast and efficient backend API.
- **Google Gemini (Chat Generative AI)**: For natural language processing (NLP) and resume analysis.
- **Markdown**: To structure the output of the AI model, which is then converted into readable HTML.
- **Bootstrap 5**: For responsive and sleek UI design.
- **Python 3.x**: The primary programming language.

## Project Screenshots :

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Resume%20Screening%20with%20AI/static/h1.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Resume%20Screening%20with%20AI/static/h2.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Resume%20Screening%20with%20AI/static/h3.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Resume%20Screening%20with%20AI/static/h4.JPG)

![Application Logo](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Resume%20Screening%20with%20AI/static/h5.JPG)

## 🧑‍💻 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MagicDash91/ML-Engineering-Project.git
   cd resume-matcher
Install Dependencies

2. **Create a virtual environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate     # For Windows
Create Virtual Environment

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
Install Libraries

4. **Set up your Google API key:**
Make sure to have a valid API key for Google Gemini. If you don’t have it, you can get it from Google Cloud Console.
In your project, replace api_key with your actual API key in the analyze_resumes function.

5. **Run the application:**
uvicorn main:app --reload
The app will be running at http://127.0.0.1:9000.

6. **Test the application:**
Visit the link above and upload a job description along with resumes (PDF or DOCX format).

The AI will analyze and provide match scores and feedback on each resume.

📝 Project Structure

resume-matcher/
│
├── main.py             # FastAPI app and routing logic
├── requirements.txt    # Required Python dependencies
├── static/             # Folder to store uploaded files
└── README.md           # This file

🤖 How It Works
Upload Resumes: Users can upload multiple resumes in PDF or DOCX format.
Job Description: Users input a job description that the resumes will be evaluated against.
Google Gemini Model: The Google Gemini API is used to analyze how well the resume matches the job description. The system scores each resume and provides a detailed explanation.
Results: The results are displayed as a clean list, where each resume is scored along with the breakdown of relevant skills and experiences.
📄 Example Results
After analysis, the application will return the results in the following format:


Candidate 1
Score: 8/10
Explanation: The candidate has relevant skills and experience in Python, Data Science, and Machine Learning. However, they are missing some specific experience with SQL databases.

Candidate 2
Score: 6/10
Explanation: The candidate has some experience with data analysis but lacks key qualifications in machine learning and data engineering.
💡 Contributing
Feel free to fork the repository, make changes, and create pull requests. Contributions are always welcome!

🤝 License
This project is licensed under the MIT License - see the LICENSE file for details.



You can copy and paste this content directly into your `README.md` file.
