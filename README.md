# hr_gpt
# Project Setup Guide

## **How to Run the Code**

1. **Update the OpenAI API Key**
   - Open `constants.py` and set your OpenAI API key:
     ```python
     openai_key = "your-api-key-here"
     ```

2. **Install Dependencies**
   - Run the following command to install required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Main Script**
   - Execute the following command:
     ```bash
     python main.py
     ```

## **How to Modify Queries**
- To change the query, edit the query variable inside `main.py`.

## **Database Configuration**
- The database files are located in the `DB` folder.
- To change databases, update the respective files in the `DB` directory.

## **Updating Documents**
- Documents are stored in the `documents` folder.
- If you need to change the document path, update the corresponding path inside `main.py`.

---

For further customization, modify the necessary files accordingly.

