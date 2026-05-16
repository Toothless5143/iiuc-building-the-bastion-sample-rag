# Secure AI Development  
## Building the BASTION.

_A Sample Project & Demo for the IIUC Cyber Security Club Webinar_

---

## Overview

This repository supports the **IIUC Cyber Security Club** webinar:  
**Secure AI Development – Building the BASTION.**

It demonstrates secure practices for AI application development with a focus on Retrieval-Augmented Generation (RAG) architectures. The code examples and guides here illustrate how to adopt security-by-design for AI pipelines using Python backend logic and HTML user interfaces.

---

## 📋 Webinar Agenda

**Phase 01: Threat Modeling & Blueprinting**

- Identifying and modeling threats to AI systems  
- Defining defense-in-depth and secure development blueprints

**Phase 02: Data Sanitization & Integrity**

- Securing data pipelines  
- Techniques for input/output validation and preserving data integrity

**Phase 04: Guardrails & RAG Architecture**

- Building effective guardrails for AI models  
- Secure design and deployment of Retrieval-Augmented Generation (RAG) systems

**Phase 05: Continuous Red Teaming**

- Implementing continuous security assessments  
- Red teaming strategies for evolving AI threats  

---

## 🖥️ Tech Stack

- **Python** (56.3%) – Backend logic, data pipelines, and security controls
- **HTML** (43.7%) – Frontend demonstration, visualization, and educational content

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or above  
- Web browser (for UI demonstration)  
- (Optional) Virtual environment tool

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Toothless5143/iiuc-building-the-bastion-sample-rag.git
   cd iiuc-building-the-bastion-sample-rag
   ```

2. **(Optional) Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

- Launch the Python backend:
  ```bash
  python app.py
  ```
app.py - usual rag
app_guardrails.py - guardrail rag

- Open the relevant URL in your browser to interact with the demo UI.

---

## 🔐 Security Best Practices Demonstrated

- Threat modeling & defense planning
- Input/output data validation and sanitization
- Secure management of secrets and environment variables
- Guardrails for AI model responses (to prevent prompt injection, data leakage, etc.)
- Continuous security testing and red teaming approaches


## 🙏 Acknowledgements

- **Webinar Host:** IIUC Cyber Security Club  
- **Project Author:** [Toothless5143](https://github.com/Toothless5143)  

_This repository is for educational and demonstration purposes as part of the “Secure AI Development: Building the BASTION.” webinar series._
