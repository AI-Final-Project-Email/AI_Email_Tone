# Email Analysis and Emotion Detection

This project provides a comprehensive solution for analyzing email content, detecting emotional tone, and generating actionable suggestions for improvement. It includes the following features:

- **Grammar and Spelling Check**: Corrects errors in email text.
- **Readability Analysis**: Evaluates the clarity and readability of the email.
- **Sentiment Analysis**: Determines whether the sentiment is positive, negative, or neutral.
- **Emotion Detection**: Identifies specific emotional states (e.g., joy, anger, sadness).
- **Improvement Suggestions**: Offers actionable recommendations for writing better emails.

## **Prerequisites**

Ensure you have the following installed on your system:

- **Python 3.10+**
- **pip** (Python package manager)

The attached notebook file should be able to be run in sequence, with the first cell checking that all necessary dependencies are installed but if you should come across a `no module named <package>` error you should be able to add that module to the cell that says `!pip install <packages>` and the notebook should run sequentially. 

There is a GPU option in the fourth part for GPU enabled systems. 

### **Dependencies**
Required Python libraries:

```bash
ipython
language_tool_python
matplotlib
nltk
pandas
spacy
textstat
torch
transformers
```

Additionally, download the required NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
```

# Usage

## **How to Run the Code**

### Clone this repository:
```bash
git clone https://github.com/AI-Final-Project-Email/AI_Email_Tone.git
cd AI_Email_Tone
```
This notebook is meant to be run through a python notebook enabled IDE such as Jupyter Notebook or Google Colab. 

Once the notebook is opened in one of these IDEs, the user should be able to run it sequentially. 

At this time, the email text must be manually entered. We suggest that you comment out the email being used for example and add your text in cell four, assigned to the variable name email_content. 

Run the main script to analyze an email by either choosing the `Run all` option under `Runtime` at the top of the screen or by pushing the triangle "play" button to the left of each cell. 

- Input your email content in cell four, to modify the email_content variable and directly process a specific email.
### Results will include:
1. Corrected grammar and spelling.
2. Readability and sentiment analysis.
3. Emotional tone detection.
4. Suggestions for improvement.
5. A rewritten version of the email based on the accepted suggestions.
## Key Functions
- `check_grammar_and_spelling`: Detects and corrects grammar/spelling errors.
- `analyze_readability`: Analyzes the clarity and readability of the original and corrected email.  
- `analyze_emotion`: Analyzes the emotional tone of the email.
- `summarize_and_suggest`: Summarizes the email and provides actionable suggestions for improvement.
## Output
**The program provides**:
1. **Visualizations**:
    - Bar charts for errors, readability, sentiment, and emotional tone.
    - Pie chart for emotion probabilities.
    - Line graph of corrected errors.
    - Pandas DataFrames to quantitatively compare original versus corrected data. 
2. **Textual Results**:
    - Detailed analysis of grammar issues, readability, sentiment, and emotion detection.
    - Suggestions for email improvements.
    - A rewritten version of the email.
## Sample Input and Output
**Sample Email**
```
Hi Team,

I hope this email find you well. I think we need to discuss the delys in our project. You alawys say the deadlines are manageble, but they never seems to be met.

Maybe we could have planned better, but I shouldn’t have to constanly follows up. No offens, but it feels like some of you arn’t taking ownership of yur tasks.

Please review your tasks before tomorow’s meeting and come prepars with suggestions for improvement.

Thanx,  
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
    - Ensure all models (e.g., facebook/bart-large-cnn, roberta-large-mnli) are correctly downloaded. This can mostly likely be corrected with `!pip install <package>`
4. Performance:
    - For GPU support, ensure PyTorch is installed and that GPU support is enabled in your IDE.