#!/usr/bin/env python

# Dependencies to be installed
get_ipython().system('pip install language-tool-python matplotlib nltk seaborn spacy textblob textstat transformers')

# Libraries to be imported
import language_tool_python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from textblob import TextBlob
import textstat
import torch
from transformers import pipeline

from IPython.display import display, HTML
import pandas as pd
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns


# ## **Part 1: Check Spelling and Grammar**

# Using **Natural Language Processing (NLP)** tools, the function analyzes the email text for linguistic issues and provides actionable corrections. By allowing users to interactively choose corrections or provide their own replacements, this tool ensures flexibility and user engagement while enhancing the overall quality and professionalism of the text.
# 
# ### **Tools Used**
# 1. **LanguageTool (via `language_tool_python`)**:
#   - An open-source grammar and style checker capable of identifying a wide range of grammatical, spelling, and stylistic issues.
#   - Provides detailed suggestions and explanations for detected errors.
# 2. **Python (Interactive Corrections)**:
#   - Implements a user-friendly interface to review and apply corrections sequentially.
#   - Dynamically adjusts the content during corrections to maintain the integrity of offsets.
# 
# ### **Key Features**
# 1. **Interactive Correction Process**:
#   - Allows the user to select suggestions or provide custom corrections.
# 2. **Error Breakdown Chart**:
#   - Displays the distribution of error types (Grammar, Spelling, Style) as a bar chart.
# 3. **Error Resolution Timeline**:
#   - Allows the user to select suggestions or provide custom corrections.

# In[5]:


def check_grammar_and_spelling(email_content):
    """
    Checks the grammar and spelling of the given email content, provides corrections,
    and generates visualizations (Error Breakdown Chart and Error Resolution Timeline).

    Args:
        email_content (str): The text of the email to analyze.

    Returns:
        dict: A dictionary with the following keys:
              - "issues" (list): A list of grammar and spelling issues detected.
              - "corrected_email" (str): The corrected version of the email.
    """
    # Initialize the LanguageTool object for English
    tool = language_tool_python.LanguageTool('en-US')

    # Detect grammar and spelling issues
    matches = tool.check(email_content)

    # Extract details of each match
    issues = []
    for match in matches:
        issues.append({
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "error_length": match.errorLength,
            "error_text": email_content[match.offset:match.offset + match.errorLength],
            "type": "Grammar" if "grammar" in match.ruleId.lower() else "Spelling" if "spelling" in match.ruleId.lower() else "Style"
        })

    # Sort matches by offset to apply corrections sequentially
    issues = sorted(issues, key=lambda x: x['offset'])

    # Interactive correction process
    print("Detected Issues:")
    corrected_content = email_content
    adjustment = 0  # Tracks cumulative changes to the offset
    correction_steps = []

    for i, issue in enumerate(issues):
      # Adjust offset for changes made to previous text
      adjusted_offset = issue['offset'] + adjustment

      # Skip if adjusted offset is invalid or overlaps corrected content
      if adjusted_offset < 0 or adjusted_offset >= len(corrected_content):
          continue

      print(f"\nError: '{issue['error_text']}' ({issue['message']})")
      if issue['suggestions']:
          print("Suggestions:")
          for j, suggestion in enumerate(issue['suggestions'], 1):
              print(f"  {j}: {suggestion}")
          print("  0: Enter a custom replacement")
      else:
          print("No suggestions available. Enter your own replacement.")

      while True:
          user_choice = input("Choose a suggestion (number) or provide your replacement: ")
          if user_choice.isdigit():
              choice_index = int(user_choice)
              if 0 <= choice_index <= len(issue['suggestions']):
                  # Apply suggestion or prompt for custom replacement
                  if choice_index == 0:
                      custom_replacement = input("Enter your replacement: ")
                  else:
                      custom_replacement = issue['suggestions'][choice_index - 1]

                  # Log step
                  correction_steps.append({
                      "step": i + 1,
                      "error": issue["error_text"],
                      "correction": custom_replacement
                  })

                  # Apply replacement and update offset adjustment
                  corrected_content = (
                      corrected_content[:adjusted_offset] +
                      custom_replacement +
                      corrected_content[adjusted_offset + issue['error_length']:]
                  )
                  adjustment += len(custom_replacement) - issue['error_length']
                  break
              else:
                  print("Invalid choice. Please enter a valid number.")
          else:
              print("Invalid input. Please enter a number.")

    # Visualization: Error Breakdown Chart
    error_types = [issue["type"] for issue in issues]
    error_counts = {etype: error_types.count(etype) for etype in set(error_types)}

    plt.figure(figsize=(8, 6))
    plt.bar(error_counts.keys(), error_counts.values(), color=["skyblue", "lightgreen", "salmon"])
    plt.title("Error Breakdown")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.show()

    # Visualization: Error Resolution Timeline
    if correction_steps:
        steps = [step["step"] for step in correction_steps]
        errors = [step["error"] for step in correction_steps]
        corrections = [step["correction"] for step in correction_steps]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, errors, label="Errors", marker="o", linestyle="--", color="lightblue")
        plt.plot(steps, corrections, label="Corrections", marker="x", linestyle="-", color="lightgreen")
        plt.title("Error Resolution Timeline")
        plt.xlabel("Step")
        plt.xticks(steps)
        plt.ylabel("Text")
        plt.legend()
        plt.grid()
        plt.show()

    # Print the final corrected content
    print("\nFinal Corrected Content:")
    print(corrected_content)

    # Return the detected issues and the corrected email
    return {
        "issues": issues,
        "corrected_email": corrected_content
    }


# In[6]:


# # Example email content
# email_content = """
# Hi team,

# I wanted to tells yall that the project's dely is causing some problems.
# We need to discusses this further. Can you gives me a update?

# Thanks,
# Jane
# """

# email_content = """
# Hi Team,

# I hope this emial finds you well. I think we needs to have an honest discuss about the delays in our projet. You alway say the deadlines are manageble, but they never seem to be met.

# Maybe we could of planned better, but I shouldn’t of have to constantly follow up. No offense, but it feel like some of you aren’t taking ownership of your task.

# I beleive we can turn this around, but I’m concern about how this might effect our client relationship. Could we perhaps revisit the task allocations and insure everyone understands their responsibilites?

# Also, just saying, it might be helpful if everyone shares there updates during tommorow’s meeting. That way, their’s no confusion about progress.

# Please review your tasks before the meeting and come prepard with any suggestion for improvement.

# Thnaks,
# Alex
# """

email_content = """
Hi John,

I hope youre doing well. I doesn't mean to be roode, but we really need to talk about the report.
You always say yo will sends it, but it never happen. I shouldnt have to remind again.
Please get it to me bye tomorow.

Best,
Mark
"""

# Analyze grammar and spelling interactively
result = check_grammar_and_spelling(email_content)
# check_grammar_and_spelling(email_content)

# # Print the final corrected content
print("\nFinal Corrected Content:")
print(result["corrected_email"])


# ## **Part 2: Clarity and Readability Check**

# Clarity and readability check is added before sentiment analysis with the understanding that enhanced clarity can help improve the accuracy of sentiment analysis by eliminating ambiguity. This is essential for reducing misunderstandings in email correspondence.
# 
# **Tools Used**
# 1. **TextStat Library**:
# - A Python library used to calculate various readability metrics.
# 
# 
# - **Key Metrics Used**:
#   -  **Flesch Reading Ease**:
#     - A higher score means that the text is easier to read.
#   - **Flesch-Kincaid Grade Level**:
#     - Represents the US school grade requied to understand the text.
#   - **Dale-Chall Readability Score**:
#     - Measures how difficult words are in a text.
#     - A lower score means easier readability.
#   - **Reading Time (minutes)**:
#     - Provides an estimated reading time based on a standard reading speed.
# 
# 2. **Pandas Library**:
# - Used to organize readability metrics into a tabular format.
# 
# 3. **Matplotlib Library**:
# - Used to create visual representations of the readability metrics.
# - **Visualizations**:
#   - **Bar Chart for Original Text**:
#     - Highlights the readability metrics of the original content.
#   - **Comparison Chart**:
#     - Displays side-by-side comparisons of metrics for the original and corrected text, showcasing improvements.

# In[7]:


def analyze_readability(original_text, corrected_text=None):
    """
    Analyze the readability of the given text and optionally compare it with corrected text.

    Args:
        original_text (str): The original text to analyze.
        corrected_text (str, optional): The corrected text to compare against the original.

    Returns:
        pd.DataFrame: A DataFrame containing readability metrics for the original and optionally corrected text.
    """
    # Calculate readability scores for the original text
    original_scores = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(original_text),
        "Grade Level": textstat.flesch_kincaid_grade(original_text),
        "Dale-Chall Readability Score": textstat.dale_chall_readability_score(original_text),
        "Reading Time (minutes)": round(len(original_text.split()) / 200, 2)  # Assuming 200 words per minute
    }

    # Initialize corrected scores as None
    corrected_scores = None

    if corrected_text:
        # Calculate readability scores for the corrected text
        corrected_scores = {
            "Flesch Reading Ease": textstat.flesch_reading_ease(corrected_text),
            "Grade Level": textstat.flesch_kincaid_grade(corrected_text),
            "Dale-Chall Readability Score": textstat.dale_chall_readability_score(corrected_text),
            "Reading Time (minutes)": round(len(corrected_text.split()) / 200, 2)
        }

        # Create a DataFrame for comparison
        df = pd.DataFrame([original_scores, corrected_scores], index=["Original", "Corrected"])
        # print("\nReadability Metrics Comparison:")
        # print(df)

        # Prepare data for comparison visualization
        labels = list(original_scores.keys())
        original_values = [original_scores[metric] for metric in labels]
        corrected_values = [corrected_scores[metric] for metric in labels]

        # Create a bar chart for readability comparison
        x = range(len(labels))
        plt.figure(figsize=(12, 6))
        plt.bar(x, original_values, width=0.4, label='Original', color='lightblue', align='center')
        plt.bar([pos + 0.4 for pos in x], corrected_values, width=0.4, label='Corrected', color='lightgreen', align='center')

        # Add chart details
        plt.title("Comparison of Readability Metrics: Original vs. Corrected")
        plt.xlabel("Readability Metrics")
        plt.ylabel("Scores")
        plt.xticks([pos + 0.2 for pos in x], labels, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        # Create a DataFrame for the original text only
        df = pd.DataFrame([original_scores], index=["Original"])
        print("\nReadability Metrics for Original Text:")
        print(df)

        # Visualization for original text only
        labels = list(original_scores.keys())
        original_values = [original_scores[metric] for metric in labels]

        plt.figure(figsize=(8, 6))
        plt.bar(labels, original_values, color="lightblue")
        plt.title("Readability Metrics for Original Text")
        plt.xlabel("Metrics")
        plt.ylabel("Scores")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return df


# In[8]:


# Compare readability scores
analyze_readability(email_content, result['corrected_email'])


# ## **Part 3: Sentiment Analysis**

# Sentiment analysis evaluates the emotional tone of email content, classifying it as **positive**, **neutral**, or **negative**. This ensures the email's tone aligns with its intended purpose, reducing miscommunication.
# 
# ### **VADER Sentiment Analysis**
# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based tool designed for short-form text, such as emails or social media posts.
# 
# **Key Features**:
# - Assigns scores for:
#   - **Positive**, **Neutral**, **Negative**, and **Compound Sentiment** (overall tone: -1 to +1).
# - Considers context, including:
#   - **Intensifiers** (e.g., "very"), **negations** (e.g., "not happy"), and punctuation or emoticons (e.g., "!!!", ":)").
# 
# **Implementation**
# - **Input**: The email content is analyzed for sentiment scores.
# - **Output**: A table and bar chart provide sentiment breakdown and overall tone classification.

# In[9]:


def analyze_sentiment(email_content):
    """
    Analyze the sentiment of the given email content.

    Args:
        email_content (str): The email content to analyze.

    Returns:
        dict: Sentiment analysis results, including scores and overall sentiment.
    """
    # Ensure the VADER lexicon is available
    nltk.download('vader_lexicon', quiet=True)

    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(email_content)

    # Determine the overall sentiment based on the compound score
    compound_score = sentiment_scores["compound"]
    if compound_score >= 0.05:
        overall_sentiment = "Positive"
    elif compound_score <= -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    # Add the overall sentiment to the results
    sentiment_results = {
        "Positive Score": sentiment_scores["pos"],
        "Neutral Score": sentiment_scores["neu"],
        "Negative Score": sentiment_scores["neg"],
        "Compound Score": sentiment_scores["compound"],
        "Overall Sentiment": overall_sentiment
    }

    # Convert sentiment results into a DataFrame
    sentiment_df = pd.DataFrame([sentiment_results])

    # Reset the index for display
    sentiment_df.index = ['']

    # Visualization: Sentiment Scores
    labels = ["Positive", "Neutral", "Negative", "Compound"]
    scores = [
        sentiment_scores["pos"],
        sentiment_scores["neu"],
        sentiment_scores["neg"],
        sentiment_scores["compound"]
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, scores, color=["lightgreen", "lightblue", "salmon", "gray"])
    plt.title("Sentiment Scores")
    plt.xlabel("Sentiment Type")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # Normalized scores
    plt.tight_layout()
    plt.show()

    return sentiment_df


# In[10]:


# Perform sentiment analysis on the corrected email content
analyze_sentiment(result["corrected_email"])


# ## **Part 4: Emotion Detection**

# ### **What Emotion Detection Adds Beyond Sentiment Analysis**
# - **Granularity**:
#   - Emotion detection identifies **specific emotional states** like joy, sadness, anger, fear, surprise, etc., offering a deeper understanding of emotional tone.
# - **Contextual Insight**:
#   - Sentiment might label a message as "negative," but emotion detection tells you why. For instance:
#     - "I'm upset about the delays" = Anger
#     - "I'm worried about the delays" = Fear
#     - Both are negative in sentiment but reflect distinct emotions.
# - **Practical Applications**:
#  - Emotion detection helps fine-tune responses. For example:
#     - If **fear** is detected, the response could aim to reassure.
#     - If **anger** is detected, the response might focus on resolving specific concerns.
# - **Additional Insights for Analysis**:
#   - Emotion detection provides a **multidimensional view**. A single text can simultaneously convey anger, sadness, and hope, which sentiment analysis might miss.
# - **Visualization and Reporting**:
#   - Emotion detection can display **multiple emotions** with varying intensities, whereas sentiment analysis focuses on a single overall label.
# 
# ### **Tools Used**
# - **spaCy (`en_core_web_sm`)**
#   - **purpose**: Linguistic Analysis
#     - Provides linguistic features such as **tokenization**, **sentence segmentation**, **part-of-speech tagging**, **named entity recognition**, and **dependency parsing**.
#   - **Useful for tasks like**:
#     - Dividing text into sentences (doc.sents).
#     - Understanding grammatical structure (e.g., detecting auxiliary verbs or adverbs for hedging).
#     - Extracting syntactic relationships between words.
# - **SentimentIntensityAnalyzer (`VADER`)**
#   - Described in **Part 3: Sentiment Analysis** as well
#   - Used in `detect_problematic_keywords`
#     - **Negative Sentences Detection**: Sentences with high negative scores are flagged as problematic.
# - **Hugging Face Pipeline (`j-hartmann/emotion-english-distilroberta-base`)**
#   - **Purpose: Emotional Tone Analysis**
#     - **What It Does**:
#       - This model detects **specific emotions** in text such as **joy, anger, fear, sadness, disgust**, and more.
#       - Provides a **probability distribution** over detected emotions, showing the likelihood of each emotion being present.
#     - **How It's Used**:
#       - `analyze_emotional_tone`:
#         - Identifies the overall emotional tone of the email.
#         - Outputs a dictionary with emotion labels and their associated probabilities.
# - **Hugging Face Pipeline (`roberta-large-mnli`)**
#   - **Purpose**: Natural Language Inference (NLI)
#     - What It Does:
#       - NLI is used to determine relationships between two pieces of text:
#         - **Entailment**: One statement logically follows from the other.
#         - **Neutral**: One statement has no strong relation to the other.
#         - **Contradiction**: One statement directly opposes the other.
#     - **How It's Used**:
#       - **Detect Hedging (`detect_hedging_nli`)**:
#         - Compares sentences (premise) with a hypothesis: `"This statement is definitive and certain."`
#         - Flags sentences classified as **neutral** with high confidence as potential hedging.
#       - **Detect Contradictions (`detect_contradictions`)**:
#         - Compares adjacent sentences in the email to find pairs flagged as **contradictory**.
# 

# In[11]:


# Detect GPU availability
device = 0 if torch.cuda.is_available() else -1


# In[12]:


# Initialize NLP models and tools
nlp = spacy.load("en_core_web_sm")  # for linguistic analysis
sia = SentimentIntensityAnalyzer()  # used in detect_problematic_words
tone_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=device)  # for emotional tone
nli_model = pipeline("text-classification", model="roberta-large-mnli", device=device) # used to detect hedging phrases and contradictory words


# In[13]:


# Function to clean and preprocess the email text
def preprocess_text(text):
    """
    Preprocess the email content to clean and standardize text.
    """
    # Remove extra spaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing spaces
    text = text.strip()
    return text

# Function to analyze emotional tone using Hugging Face model
def analyze_emotional_tone(email_content):
    result = tone_analyzer(email_content)
    return result[0] if result else None

# Function to detect indirect or passive-aggressive tone
def detect_passive_aggressive(email_content):
    """
    Detect passive-aggressive tone using a combination of hedging and contradictions.

    Args:
        email_content (str): The email content to analyze.

    Returns:
        list: A list of flagged sentences with passive-aggressive tone.
    """
    hedging = detect_hedging_nli(email_content)
    contradictions = detect_contradictions(email_content)

    return {"Hedging": hedging, "Contradictions": contradictions}

def detect_hedging_nli(email_content):
    """
    Use an NLI model to detect hedging phrases.

    Args:
        email_content (str): The email content to analyze.

    Returns:
        list: A list of sentences flagged as hedging and the count.
    """
    doc = nlp(email_content)
    hedging_phrases = []

    for sent in doc.sents:
        premise = sent.text.strip()  # Clean whitespace from sentences
        if not premise:  # Skip empty sentences
            continue

        hypothesis = "This statement is definitive and certain."
        result = nli_model(f"{premise} {hypothesis}")
        for res in result:
            if res["label"] == "NEUTRAL" and res["score"] > 0.6:
                hedging_phrases.append(sent.text.strip())

    return {"hedging_phrases": list(set(hedging_phrases)), "count": len(set(hedging_phrases))}


def detect_contradictions(email_content):
    """
    Detect contradictions using sentence pairs.

    Args:
        email_content (str): The email content to analyze.

    Returns:
        list: A list of detected contradictions.
    """
    doc = nlp(email_content)
    contradictions = []

    sentences = [sent.text for sent in doc.sents]
    for i in range(len(sentences) - 1):  # Compare sentence pairs
        pair_result = nli_model(f"{sentences[i]} {sentences[i + 1]}")
        for res in pair_result:
            if res["label"] == "CONTRADICTION" and res["score"] > 0.6:
                contradictions.append((sentences[i], sentences[i + 1]))

    return contradictions

def detect_problematic_keywords(email_content):
    """
    Detect problematic keywords dynamically using sentence-level sentiment analysis.

    Args:
        email_content (str): The email content to analyze.

    Returns:
        list: A list of sentences flagged as problematic.
    """
    doc = nlp(email_content)
    problematic_sentences = []

    for sent in doc.sents:
        sentiment = sia.polarity_scores(sent.text)
        if sentiment["neg"] > 0.5:  # Threshold for high negativity
            problematic_sentences.append(sent.text)

    return list(set(problematic_sentences))  # Unique problematic sentences

# Function to provide suggestions based on analysis
def provide_suggestions(email_content):
    suggestions = []
    if detect_passive_aggressive(email_content):
        suggestions.append("Avoid passive-aggressive phrases. Try to be more direct.")
    if len(detect_problematic_keywords(email_content)) > 0:
        suggestions.append("There are some strong words that might be misinterpreted. Consider using softer language.")
    readability_score = textstat.flesch_reading_ease(email_content)
    if readability_score < 60:
        suggestions.append("The email might be difficult to understand. Try simplifying the language.")
    return suggestions

def visualize_emotional_tone(results):
    """
    Visualize the results of emotional tone analysis, including counts and details.

    Args:
        results (dict): The analysis results, including hedging phrases, contradictions, and problematic sentences.
    """
    # Extract flagged content and counts
    hedging = results.get("Hedging Phrases", {}).get("hedging_phrases", [])
    hedging_count = results.get("Hedging Phrases", {}).get("count", 0)
    contradictions = results.get("Contradictions", [])
    problematic_sentences = results.get("Problematic Sentences", [])

    # Categories and counts for visualization
    categories = ["Hedging", "Contradictions", "Problematic Sentences"]
    counts = [hedging_count, len(contradictions), len(problematic_sentences)]

    # Visualization: Bar Chart of Detected Issues
    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color=["lightblue", "lightgreen", "salmon"])
    plt.title("Emotional Tone Analysis")
    plt.ylabel("Count")
    plt.xlabel("Categories")
    plt.ylim(0, max(counts) + 1)
    plt.tight_layout()
    plt.show()

    # Create a detailed table for flagged content
    detailed_content = (
        [{"Category": "Hedging", "Flagged Content": phrase} for phrase in hedging] +
        [{"Category": "Contradictions", "Flagged Content": f"{pair[0]} <-> {pair[1]}"} for pair in contradictions] +
        [{"Category": "Problematic Sentences", "Flagged Content": sentence} for sentence in problematic_sentences]
    )

    # Convert to DataFrame for display
    if detailed_content:
        details_df = pd.DataFrame(detailed_content)
        print("\nDetailed Results:")
        display(details_df)
    else:
        print("No flagged content found.")


# In[14]:


def analyze_emotion(email_content):
    """
    Perform a comprehensive analysis of the emotional tone of the email content,
    including an overall emotion summary, hedging, contradictions, problematic
    sentences, and suggestions.

    Args:
        email_content (str): The email content to analyze.

    Returns:
        dict: A dictionary containing analysis results, including overall emotion,
              hedging phrases, contradictions, problematic sentences, and suggestions.
        pd.DataFrame: A DataFrame summarizing the analysis results.
    """
    # Adjust pandas display options for better visualization
    pd.set_option('display.max_colwidth', None)

    # Step 1: Preprocess the email content
    cleaned_email = preprocess_text(email_content)

    # Step 2: Analyze overall emotion using a Hugging Face emotion detection model
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    emotion_results = emotion_classifier(cleaned_email)
    overall_emotion = max(emotion_results, key=lambda x: x['score'])  # Most probable emotion

    # Step 3: Detect hedging phrases using NLI
    hedging_results = detect_hedging_nli(cleaned_email)

    # Step 4: Detect contradictions between sentences
    contradictions = detect_contradictions(cleaned_email)

    # Step 5: Detect problematic sentences based on sentiment analysis
    problematic_sentences = detect_problematic_keywords(cleaned_email)

    # Step 6: Generate suggestions for improvement
    suggestions = provide_suggestions(cleaned_email)

    # Compile the results
    results = {
        "Overall Emotion": overall_emotion["label"],
        "Emotion Probabilities": {emo["label"]: emo["score"] for emo in emotion_results},
        "Hedging Phrases": hedging_results,
        "Contradictions": contradictions,
        "Problematic Sentences": problematic_sentences,
        "Suggestions": suggestions,
    }

    # Create a summary DataFrame
    data = {
        "Category": ["Overall Emotion", "Emotion Probabilities", "Hedging Phrases", "Contradictions", "Problematic Sentences", "Suggestions"],
        "Details": [
            results["Overall Emotion"],
            results["Emotion Probabilities"],
            ", ".join(results["Hedging Phrases"]["hedging_phrases"]) if results["Hedging Phrases"]["hedging_phrases"] else "None",
            ", ".join([f"{pair[0]} <-> {pair[1]}" for pair in results["Contradictions"]]) if results["Contradictions"] else "None",
            ", ".join(results["Problematic Sentences"]) if results["Problematic Sentences"] else "None",
            "; ".join(results["Suggestions"]) if results["Suggestions"] else "None",
        ]
    }

    results_df = pd.DataFrame(data)

    # Automatically display the DataFrame
    # print("\nAnalysis Summary:")
    display(results_df)

    # Visualize results
    visualize_emotional_tone(results)
    visualize_emotions_pie_chart(results["Emotion Probabilities"])



def visualize_emotions_pie_chart(emotion_probabilities):
    """
    Visualize the distribution of emotion probabilities as a pie chart.

    Args:
        emotion_probabilities (dict): A dictionary of emotion labels and their probabilities.
    """
    labels = list(emotion_probabilities.keys())
    sizes = list(emotion_probabilities.values())
    colors = ["lightblue", "lightgreen", "salmon", "orange", "purple", "pink"]  # Add more if necessary

    plt.figure(figsize=(8, 6))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
    )
    plt.title("Emotion Distribution")
    plt.axis


# In[15]:


# # Example email input
# email_content = """
# Hi John,

# I hope youre doing well. I doesn't mean to be roode, but we really need talks about the report.
# You alwyas say you will send it, but it never happen. I shouldnot have to remind you again.
# Please get it to me by tomorrow. I am very upset.

# Best,
# Mark
# """


# In[16]:


# Analyze emotional tone
emotional_intent_results = analyze_emotion(result["corrected_email"])
emotional_intent_results


# ## Part 5: Total Analysis and Suggestion of a Refined Option

# In[17]:


summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)  # for summarization


# In[18]:



# Function for summary generation with randomness
def generate_summary(email_content, max_length=30, min_length=10, num_return_sequences=3):
    """
    Generate a summary for the email content, optionally selecting a random one from multiple outputs.

    Args:
        email_content (str): The email content to summarize.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.
        num_return_sequences (int): The number of summaries to generate for randomness.

    Returns:
        str: A randomly selected summary.
    """
    # Generate multiple summaries using sampling
    summaries = summarizer(
        email_content,
        max_length=max_length,
        min_length=min_length,
        do_sample=True,  # Enable randomness in the generation process
        num_return_sequences=num_return_sequences  # Generate multiple summaries
    )

    # Randomly select one of the summaries
    chosen_summary = random.choice(summaries)["summary_text"]
    return chosen_summary


# In[19]:


# # Function to generate a better email
# def generate_better_email(email_content, suggestions):
#     """
#     Generate a rewritten version of the email based on suggestions.

#     Args:
#         email_content (str): The original email content.
#         results (dict): Analysis results containing flagged issues and suggestions.

#     Returns:
#         str: A rewritten version of the email.
#     """
#     # Modify email content iteratively based on suggestions
#     improved_email = email_content

#     for suggestion in suggestions:
#         if "Problematic:" in suggestion:
#             # Replace problematic sentence with a placeholder (real implementation can use AI to rewrite)
#             problematic_sentence = suggestion.split("Problematic: '")[1].rstrip("'")
#             improved_email = improved_email.replace(problematic_sentence, "[Revised sentence]")

#         if "Hedging:" in suggestion:
#             # Replace hedging phrases with more direct alternatives (can be AI-powered in real use cases)
#             hedging_phrase = suggestion.split("Hedging: '")[1].rstrip("'")
#             improved_email = improved_email.replace(hedging_phrase, "[Direct phrase]")

#     return improved_email
# Function to generate a better email
def generate_better_email(email_content, results):
    """
    Generate a rewritten version of the email based on analysis results.

    Args:
        email_content (str): The original email content.
        results (dict): Analysis results containing flagged issues and suggestions.

    Returns:
        str: A rewritten version of the email.
    """
    improved_email = email_content

    # Handle Hedging Phrases
    hedging_phrases = results["Hedging Phrases"]["hedging_phrases"]
    for phrase in hedging_phrases:
        improved_email = improved_email.replace(phrase, "[Direct alternative for: '{}']".format(phrase))

    # Handle Contradictions
    contradictions = results["Contradictions"]
    for pair in contradictions:
        original_text = f"{pair[0]} <-> {pair[1]}"
        improved_email = improved_email.replace(pair[0], "[Clarified statement for: '{}']".format(pair[0]))
        improved_email = improved_email.replace(pair[1], "[Clarified statement for: '{}']".format(pair[1]))

    # Handle Problematic Sentences
    problematic_sentences = results["Problematic Sentences"]
    for sentence in problematic_sentences:
        improved_email = improved_email.replace(sentence, "[Softer alternative for: '{}']".format(sentence))

    return improved_email


# # Function to summarize and suggest improvements
# def summarize_and_suggest(email_content, results):
#     """
#     Summarize the email, provide suggestions, and generate a better version.

#     Args:
#         email_content (str): The original email content.
#         results (dict): Analysis results from previous parts.

#     Returns:
#         dict: A dictionary containing the summary, suggestions, and improved email.
#     """
#     # Step 1: Generate a summary using your custom `generate_summary` function
#     summary = generate_summary(email_content, max_length=50, min_length=20, num_return_sequences=3)

#     # Step 2: Generate suggestions using your custom `provide_suggestions` function
#     suggestions = provide_suggestions(email_content)

#     # Step 3: Generate a better email based on suggestions
#     better_email = generate_better_email(email_content, suggestions)

#     # Compile results
#     return {
#         "Summary": summary,
#         "Suggestions": suggestions,
#         "Improved Email": better_email
#     }

def summarize_and_suggest(email_content, results):
    """
    Summarize the email, provide suggestions, and generate a better version.

    Args:
        email_content (str): The original email content.
        results (dict): Analysis results from Part 4.

    Returns:
        dict: A dictionary containing the summary, suggestions, and improved email.
    """
    # Step 1: Generate a summary using `generate_summary`
    summary = generate_summary(email_content, max_length=50, min_length=20, num_return_sequences=3)

    # Step 2: Extract suggestions from Part 4 analysis
    suggestions = []
    if results["Hedging Phrases"]["hedging_phrases"]:
        suggestions.append(f"Hedging detected: {', '.join(results['Hedging Phrases']['hedging_phrases'])}. Consider revising for clarity.")
    if results["Contradictions"]:
        contradictions = [f"{pair[0]} <-> {pair[1]}" for pair in results["Contradictions"]]
        suggestions.append(f"Contradictions detected: {', '.join(contradictions)}. Clarify conflicting statements.")
    if results["Problematic Sentences"]:
        suggestions.append(f"Problematic sentences detected: {', '.join(results['Problematic Sentences'])}. Use softer or more neutral language.")
    if not suggestions:
        suggestions.append("No major issues detected. Email looks fine!")

    # Step 3: Generate a better email based on Part 4 analysis
    better_email = generate_better_email(email_content, results)

    # Compile results
    return {
        "Summary": summary,
        "Suggestions": suggestions,
        "Improved Email": better_email
    }


# In[21]:


print("Emotional Intent Results:", emotional_intent_results)


# In[20]:


# Run Part 4 (Emotion Detection)
emotional_intent_results = analyze_emotion(result["corrected_email"])

# Run Part 5 (Summarization and Suggestions)
summary_and_suggestions = summarize_and_suggest(result["corrected_email"], emotional_intent_results)

# Display results
print("Summary:")
print(summary_and_suggestions["Summary"])

print("\nSuggestions:")
for suggestion in summary_and_suggestions["Suggestions"]:
    print("-", suggestion)

print("\nImproved Email:")
print(summary_and_suggestions["Improved Email"])


# In[ ]:


# # Assume `results` is the dictionary of results from Part 4 (Emotion Detection)
# summary_and_suggestions = summarize_and_suggest(result["corrected_email"], emotional_intent_results)
# summary_and_suggestions


# In[ ]:


# # Print the outputs
# print("Summary:")
# print(summary_and_suggestions["Summary"])

# print("\nSuggestions:")
# for suggestion in summary_and_suggestions["Suggestions"]:
#     print("-", suggestion)

# print("\nImproved Email:")
# print(summary_and_suggestions["Improved Email"])

