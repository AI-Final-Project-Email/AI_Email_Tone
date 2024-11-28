# Email Analysis and Emotion Detection

This project provides a comprehensive solution for analyzing email content, detecting emotional tone, and generating actionable suggestions for improvement. It includes the following features:

- **Grammar and Spelling Check**: Corrects errors in email text.
- **Readability Analysis**: Evaluates the clarity and readability of the email.
- **Sentiment Analysis**: Determines whether the sentiment is positive, negative, or neutral.
- **Emotion Detection**: Identifies specific emotional states (e.g., joy, anger, sadness).
- **Summary Generation**: Provides a concise summary of the email content.
- **Improvement Suggestions**: Offers actionable recommendations for writing better emails.

## **Prerequisites**

Ensure you have the following installed on your system:

- **Python 3.7+**
- **pip** (Python package manager)

### **Dependencies**
Install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```
The `requirements.txt` file should include:
```
transformers
spacy
language-tool-python
textstat
pandas
matplotlib
nltk
sentence-transformers
```
Additionally, download the required NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
```
For spaCy, ensure the language model is installed:
```python
python -m spacy download en_core_web_sm
```
# Usage

## **How to Run the Code**

### Clone this repository:
```bash
git clone https://github.com/your-repo/email-analysis.git
cd email-analysis
```
Run the main script to analyze an email:
```bash
python main.py
```
- Input your email content when prompted, or modify the email_content variable in main.py to directly process a specific email.
### Results will include:
1. Corrected grammar and spelling.
2. Readability and sentiment analysis.
3. Emotional tone detection.
4. Suggestions for improvement.
5. A rewritten version of the email based on the suggestions.
## Key Functions
- `check_grammar_and_spelling`: Detects and corrects grammar/spelling errors.
- `analyze_emotion`: Analyzes the emotional tone of the email.
- `summarize_and_suggest`: Summarizes the email and provides actionable suggestions for improvement.
## Output
**The program provides**:
1. **Visualizations**:
    - Bar charts for readability and emotional tone.
    - Pie chart for emotion probabilities.
2. **Textual Results**:
    - Detailed analysis of grammar issues, readability, sentiment, and emotion detection.
    - Suggestions for email improvements.
    - A rewritten version of the email.
## Sample Input and Output
**Sample Email**
```
Hi Team,

I hope this email finds you well. I think we need to discuss the delays in our project. You always say the deadlines are manageable, but they never seem to be met.

Maybe we could have planned better, but I shouldn’t have to constantly follow up. No offense, but it feels like some of you aren’t taking ownership of your tasks.

Please review your tasks before tomorrow’s meeting and come prepared with suggestions for improvement.

Thanks,  
Alex
```
## Output
- **Summary**:
`"Discuss project delays and plan improvements."`

- **Suggestions**:
    - "Avoid passive-aggressive phrases. Try to be more direct."
    - "Consider revising hedging phrases for clarity."
- **Rewritten Email**:
```
Hi Team,

I hope this email finds you well. Let's discuss the delays in our project and address any challenges. It's crucial that we meet our deadlines and collaborate effectively.

Please review your tasks before tomorrow's meeting and bring suggestions for improvement.

Best regards,  
Alex
```
## Troubleshooting
1. Model Not Found Error:
    - Ensure all models (e.g., facebook/bart-large-cnn, roberta-large-mnli) are correctly downloaded.
2. Dependencies Issue:
    - Recheck the installed libraries with pip list and match them against requirements.txt.
3. Performance:
    - For GPU support, ensure PyTorch is installed with CUDA.