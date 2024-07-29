
import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Function to batch responses into manageable groups for processing
def batch_responses(responses, batch_size):
    batches = []
    for i in range(0, len(responses), batch_size):
        batch = responses[i:i+batch_size]
        batches.append(batch)
    return batches

# Function to clean and parse JSON output
def clean_and_parse_json(json_text):
    try:
        cleaned_text = json_text.strip().strip('').strip().strip('json').strip().strip('JSON').strip()
        if cleaned_text.startswith("JSON") or cleaned_text.startswith("json"):
            cleaned_text = cleaned_text[4:].strip()
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error: {e}")
        st.error(f"Original text: {json_text}")
        return []

# Generate survey response analysis and parse the output into a DataFrame
def generate(survey_question, responses, batch_size, model):
    prefix = """
    Analyze the given survey responses based on the following criteria:

    Step 1: Relevance Analysis: Classify responses as 'Relevant' unless they add no value or are unrelated to the question. Assess with an inclination towards finding relevance:
    - Direct Address: Does the response engage with the primary issue, even indirectly? Favor responses addressing any aspect of the question.
    - Contextual Relevance: Does the response provide insights, context, or perspectives related to the question? Classify as 'Relevant' if it contributes to a broader understanding.
    - Insightful Contribution: Does the response deepen understanding, provide novel viewpoints, or enrich the discussion? Consider thoughtful or analytically deep responses relevant.
    - Common Terms: Responses with even one term common with the question must be classified as 'Relevant.'
    - Open-Ended Responses: Responses like 'none,' 'nothing,' 'not sure,' or 'I am satisfied' should be classified as 'Relevant' as they directly address the question.

    Step 2: AI Generation Likelihood: Evaluate the probability that the response was generated by an AI. Look for indicators like:
    - Excessive use of adjectives.
    - Phrasing that is atypical of normal human responses, such as speaking like an eloquent expert.
    - Overly structured responses.
    - Absence of grammatical mistakes.
    - Use of words like "ignite innovation" and "delve".

    Step 3: Output your analysis in a JSON array with the following structure:
    {
        "response": "The verbatim survey response.",
        "relevance": "Indicate 'Relevant' if the response adds any value or is related to the question. Mark 'Non-Relevant' only if it is completely devoid of content or context.",
        "relevancy_rating": "Assign a rating from 0 (no relevance) to 10 (high relevance).",
        "ai_generation_likelihood": "'Likely AI-Generated' or 'Non AI-Generated'."
    }

    IMPORTANT: Analyze all responses provided, regardless of content quality (including duplicates or responses like n/a, none, or nothing). Ensure the number of input responses equals the number of output responses.
    """

    batches = batch_responses(responses, batch_size)
    all_parsed_data = []

    start_time = time.time()

    for i, batch in enumerate(batches):
        prompt = f"""
        {prefix}

        Survey Question: {survey_question}

        Responses:
        {json.dumps(batch)}

        Analyze these responses according to the given criteria. Ensure the output is in JSON format.
        """
        result = model.generate_content(prompt)
        st.write(f"Batch {i + 1} response:\n{result.text}\n")  # Print the model's response for each batch
        
        try:
            parsed_output = clean_and_parse_json(result.text)
            all_parsed_data.extend(parsed_output)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON for batch {i + 1}: {e}")
            st.error(f"Original text: {result.text}")
        
        # Update progress bar
        progress = (i + 1) / len(batches)
        progress_bar.progress(progress, f"{progress:.1%}")
        
        time.sleep(10)  # Avoid hitting rate limits

    # Convert parsed data to DataFrame
    df = pd.DataFrame(all_parsed_data)

    end_time = time.time()
    st.sidebar.write(f"Total Code Running Time: {end_time - start_time:.2f} seconds")
    return df

# Function to create word cloud
def create_word_cloud(responses):
    text = ' '.join(responses)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to create violin plot
def create_violin_plot(df):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='relevance', y='relevancy_rating', data=df, inner='stick')
    plt.title('Distribution of Relevancy Ratings')
    plt.xlabel('Relevance')
    plt.ylabel('Relevancy Rating')
    st.pyplot(plt)

# Function to cluster responses using the provided similarity algorithm
def cluster_responses(df, threshold):
    # Preprocess the responses
    df['response'] = df['response'].astype(str).str.lower()

    # Vectorize the responses
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(df['response'])
    except ValueError:
        st.error("Error: Empty vocabulary. The responses may only contain stop words or be empty.")
        return df

    # Compute the similarity matrix
    similarity_matrix = cosine_similarity(X)

    # Compute the distance matrix
    distance_matrix = 1 - similarity_matrix

    # Ensure all distances are non-negative
    distance_matrix = np.maximum(distance_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)

    # Convert to condensed distance matrix
    condensed_distance_matrix = squareform(distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method='average')

    # Form clusters based on the specified threshold
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')

    # Assign clusters to the DataFrame
    df['Group'] = clusters

    # Calculate similarity scores based on maximum similarity to other responses in the same group
    similarity_scores = []
    for i, group in enumerate(clusters):
        same_group_indices = [j for j, g in enumerate(clusters) if g == group and j != i]
        if same_group_indices:
            max_similarity = max(similarity_matrix[i, same_group_indices])
            similarity_scores.append(max_similarity)
        else:
            similarity_scores.append(np.nan)

    df['Similarity Score'] = similarity_scores

    # Identify unique groups and update the DataFrame
    group_counts = df['Group'].value_counts()
    unique_groups = group_counts[group_counts == 1].index

    df.loc[df['Group'].isin(unique_groups), ['Group', 'Similarity Score']] = ""

    return df

# Streamlit interface
col1, col2 = st.columns([1, 3])
with col1:
    st.image("ppl_logo.jpg", width=100)  # Replace with your image path
with col2:
    st.markdown("<h1 style='text-align: left; color: lightblue;'>Phronesis Apex : Mini</h1>", unsafe_allow_html=True)

# Sidebar for API key and settings
st.sidebar.title("API Settings")
api_key = st.sidebar.text_input('Gemini API Key', type='password')
batch_size = st.sidebar.number_input('Batch Size', min_value=1, value=15)
model_name = st.sidebar.selectbox('Select Model', ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro'])

# Configure the API key and model
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
else:
    model = None

# Store responses in session state
if 'responses' not in st.session_state:
    st.session_state['responses'] = []
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'df_clustered' not in st.session_state:
    st.session_state['df_clustered'] = None
if 'survey_question' not in st.session_state:
    st.session_state['survey_question'] = ''

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Submit Open Ends", "AI Analysis", "Visualizations", "Similarity Check", "History"])

with tab1:
    st.info("Please add both the survey question and survey responses. After that, you can proceed with analysis or similarity checks in the respective tabs.")
    # User input for survey question and responses
    survey_question = st.text_area('Survey Question')
    responses_text = st.text_area('Survey Responses (one per line)')

    if st.button('Submit Responses'):
        if survey_question and responses_text:
            responses = responses_text.split('\n')
            st.session_state['responses'] = responses
            st.session_state['survey_question'] = survey_question
            st.success("Responses submitted successfully!")
        else:
            st.error('Please fill in both the survey question and responses.')

with tab2:
    if 'responses' in st.session_state and st.session_state['responses']:
        if st.button('Analyze Responses'):
            if model is not None:
                responses = st.session_state['responses']
                survey_question = st.session_state['survey_question']
                st.write(f"Total number of responses: {len(responses)}")

                # Progress bar
                progress_bar = st.progress(0)

                df = generate(survey_question, responses, batch_size, model)
                st.dataframe(df)
                st.session_state['df'] = df

                # Provide download link for the DataFrame
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "responses_analysis.csv", "text/csv", key='download-csv')
            else:
                st.error("Please enter your API key in the sidebar before analyzing responses. \n \n Go to https://aistudio.google.com/app/apikey to create your API Key.")
    else:
        st.info("Please submit responses in the Submit Responses tab first.")

with tab3:
    if 'df' in st.session_state and st.session_state['df'] is not None:
        st.subheader("Word Cloud of Responses")
        create_word_cloud(st.session_state['responses'])
        
        st.subheader("Violin Plot of Relevancy Ratings")
        create_violin_plot(st.session_state['df'])
    else:
        st.info("Please analyze the responses in the Analysis tab first.")

with tab4:
    st.subheader("Similarity Check")
    if 'responses' in st.session_state and st.session_state['responses']:
        if model is not None:
            threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.3, 0.1)
            df_similarity = pd.DataFrame({'response': st.session_state['responses']})
            df_clustered = cluster_responses(df_similarity, threshold)
            st.dataframe(df_clustered)
            st.session_state['df_clustered'] = df_clustered
        else:
            st.error("Please enter your API key in the sidebar before running the similarity check.")
    else:
        st.info("Please submit responses in the Submit Responses tab first.")

with tab5:
    st.subheader("History")
    if 'df' in st.session_state and st.session_state['df'] is not None:
        st.write("Analysis Data:")
        st.dataframe(st.session_state['df'])
    if 'df_clustered' in st.session_state and st.session_state['df_clustered'] is not None:
        st.write("Similarity Data:")
        st.dataframe(st.session_state['df_clustered'])

    # if 'responses' in st.session_state and st.session_state['responses']:
    #     st.write("Raw Responses:")
    #     st.dataframe(pd.DataFrame(st.session_state['responses'], columns=['Response']))
    else:
        st.info("No data available. Please run the analysis or similarity check first.")
