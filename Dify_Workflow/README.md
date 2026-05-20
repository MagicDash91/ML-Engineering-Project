# Dify AI Workflow Projects

Two production-ready AI workflow projects built on **Dify** using **Gemini 2.5 Flash**, demonstrating real-world agentic patterns including document analysis, OCR, career intelligence, and automated email delivery.

---

## Projects Overview

| Project         | Type                      | Model            | Output                              |
| --------------- | ------------------------- | ---------------- | ----------------------------------- |
| **CV Analyzer** | Document Analysis + Email | Gemini 2.5 Flash | CV Score Report via Email           |
| **Agentic AI**  | Multi-class Intent Router | Gemini 2.5 Flash | OCR / CV Analysis / Career Guidance |

---

## Project 1 — CV Analyzer

### Description

An automated CV scoring system that analyzes a candidate's resume and delivers a structured evaluation report directly to their email. Built for recruitment automation and career development use cases.

## Screenshots

![Application Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Dify_Workflow/static/a1.JPG)

![Application Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Dify_Workflow/static/a2.JPG)

![Application Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Dify_Workflow/static/a3.JPG)

---

### Architecture

```
USER INPUT (CV file + Email Address)
      ↓
LLM (Gemini 2.5 Flash)
— Analyzes CV across 7 dimensions
— Generates structured markdown report
      ↓
HTTP REQUEST (Resend API)
— Delivers formatted report to candidate email
      ↓
OUTPUT
```

### Features

- Upload CV in PDF format
- Automatic scoring across 7 sections (Contact, Summary, Experience, Education, Skills, Achievements, Presentation)
- Total score out of 70
- Strengths and improvement areas per section
- Top 3 actionable recommendations
- Automated email delivery via Resend

### Tech Stack

- **Platform**: Dify Cloud
- **LLM**: Gemini 2.5 Flash
- **Email**: Resend API
- **Input formats**: PDF
- **Output**: Markdown report delivered to email

### Workflow Nodes

```
USER INPUT → LLM → OUTPUT
                ↘
                HTTP REQUEST (Resend email)
```

### Sample Output

```
**CV SCORING REPORT**

1. **Contact & Personal Info** — 9/10
   - Strengths: Clear, professional contact details
   - Improvements: Add LinkedIn vanity URL

2. **Professional Summary** — 0/10
   - Strengths: N/A
   - Improvements: Add 3-5 sentence summary

3. **Work Experience** — 8/10
   - Strengths: Quantifiable achievements, strong action verbs
   - Improvements: Clarify overlapping dates

...

**Total Score: 52/70**

**Top 3 Recommendations:**
1. Add a concise professional summary immediately
2. Condense CV to 2 pages maximum
3. Move LinkedIn and GitHub to top of document
```

### Environment Variables

```
GOOGLE_API_KEY=your_gemini_api_key
RESEND_API_KEY=your_resend_api_key
```

---

## Project 2 — Agentic AI

### Description

A multi-class intelligent routing system that automatically detects the user's intent and routes their question to the appropriate specialized AI agent. Demonstrates agentic patterns using Dify's Question Classifier node as the core orchestration mechanism.

## Screenshots

![Application Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Dify_Workflow/static/a4.JPG)

![Application Screenshot 1](https://raw.githubusercontent.com/MagicDash91/ML-Engineering-Project/main/Dify_Workflow/static/a5.JPG)

---

### Architecture

```
USER INPUT (question + optional file)
      ↓
QUESTION CLASSIFIER (Gemini 2.5 Flash)
— Detects intent from natural language
— Routes to appropriate specialist
      ↓
┌─────────────────────────────────┐
│  CLASS 1       CLASS 2       CLASS 3  │
│  OCR Agent  CV Analyzer  Career Coach │
└─────────────────────────────────┘
      ↓
VARIABLE AGGREGATOR
— Collects output from whichever branch ran
      ↓
OUTPUT
```

### Intent Classes

| Class                         | Trigger                                                                               | Specialist Agent          |
| ----------------------------- | ------------------------------------------------------------------------------------- | ------------------------- |
| **CLASS 1 — OCR**             | User wants to extract text from image, scan KTP, passport, or any document            | LLM OCR Agent             |
| **CLASS 2 — CV Analysis**     | User wants CV scoring, resume feedback, or application review                         | LLM CV Analyzer Agent     |
| **CLASS 3 — Career Guidance** | User asks for career advice, path planning, skill development, or professional growth | LLM Career Guidance Agent |

### Features

- Natural language intent detection (no buttons, no menus)
- 3 specialized AI agents with domain-specific prompts
- Handles both text questions and file uploads
- Single clean output regardless of which branch runs
- Extensible — add more classes without changing downstream nodes

### Tech Stack

- **Platform**: Dify Cloud
- **LLM**: Gemini 2.5 Flash (all nodes)
- **Core Node**: Question Classifier
- **Input**: Text question + optional file (PDF, image, document)
- **Output**: Structured text response

### Workflow Nodes

```
USER INPUT
      ↓
QUESTION CLASSIFIER
   ├── CLASS 1 → LLM OCR ──────────────┐
   ├── CLASS 2 → LLM CV ANALYZER ──────┤→ VARIABLE AGGREGATOR → OUTPUT
   └── CLASS 3 → LLM CAREER GUIDANCE ──┘
```

### Agent Prompts

**OCR Agent**

```
You are an expert document scanner and OCR specialist.
Extract all text from the provided image or document
accurately and completely. Preserve the original structure.
Output ONLY the extracted text, nothing else.
```

**CV Analyzer Agent**

```
You are an expert HR consultant and CV reviewer
specializing in Indonesian tech industry standards.
Analyze CVs objectively with actionable feedback.
Score each section out of 10 with strengths and improvements.
```

**Career Guidance Agent**

```
You are an expert career coach specializing in
tech professionals in Indonesia. You have deep
knowledge of the Indonesian job market, career
progression paths in AI/ML and software engineering,
and practical guidance for career growth.
You give honest, actionable, personalized advice.
```

### Sample Interactions

**OCR:**

> Input: _"Can you read this KTP?"_ + KTP image
> Output: Extracted text with all identity fields

**CV Analysis:**

> Input: _"Can you score my CV?"_ + PDF resume
> Output: Structured scoring report with recommendations

**Career Guidance:**

> Input: _"How should I grow my career as an AI engineer in Indonesia?"_
> Output: Structured career roadmap with skills, timeline, and resources

---

## Key Concepts Demonstrated

| Concept                   | Project     | Implementation                   |
| ------------------------- | ----------- | -------------------------------- |
| **Document Analysis**     | CV Analyzer | PDF parsing + LLM scoring        |
| **Email Automation**      | CV Analyzer | Resend API via HTTP Request node |
| **Intent Classification** | Agentic AI  | Question Classifier node         |
| **Dynamic Routing**       | Agentic AI  | Class-based branching            |
| **Output Aggregation**    | Agentic AI  | Variable Aggregator node         |
| **Multi-modal Input**     | Both        | Text + file upload handling      |
| **Prompt Engineering**    | Both        | Role-based system prompts        |

---

## What I Learned

- Building no-code/low-code agentic workflows on Dify
- Configuring Question Classifier for intent-based routing
- Integrating third-party APIs (Resend) via HTTP Request nodes
- Handling multi-modal inputs (text + PDF + images) in LLM workflows
- Using Variable Aggregator to merge parallel branch outputs
- Prompt engineering for specialized domain agents (OCR, HR, Career)
- Deploying and publishing workflows for real-world use

---

## Author

**Michael Wiryaseputra**
AI/ML Engineer | Bootcamp Trainer | MC & AI Speaker
[LinkedIn](https://linkedin.com/in/michael-wiryaseputra) | [GitHub](https://github.com/MagicDash91)

---

## Tools & Platforms

![Dify](https://img.shields.io/badge/Dify-Cloud-blue)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-green)
![Resend](https://img.shields.io/badge/Resend-Email_API-orange)
