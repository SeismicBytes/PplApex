import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
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
import re
import vertexai.preview.generative_models as generative_models

# Function to batch responses into manageable groups for processing
def batch_responses(responses, batch_size):
    return [responses[i:i+batch_size] for i in range(0, len(responses), batch_size)]

# Function to clean and parse JSON output
def clean_and_parse_json(json_text):
    try:
        # Remove code block markers if present
        cleaned_text = re.sub(r'```json\s*|\s*```', '', json_text)
        
        # Remove any leading/trailing whitespace and common prefixes
        cleaned_text = cleaned_text.strip().lstrip('json').lstrip('JSON').strip()
        
        # Attempt to parse the JSON
        parsed_json = json.loads(cleaned_text)
        
        # Ensure the result is a list
        if isinstance(parsed_json, dict):
            parsed_json = [parsed_json]
        
        return parsed_json
    except json.JSONDecodeError as e:
        st.error(f"JSON Decode Error: {e}")
        st.error(f"Original text: {json_text}")
        # Attempt to fix common JSON errors
        try:
            fixed_text = cleaned_text.replace("'", '"')  # Replace single quotes with double quotes
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)  # Add quotes to keys
            return json.loads(fixed_text)
        except:
            st.error("Could not parse JSON even after attempted fixes.")
            return []

# Generate survey response analysis and parse the output into a DataFrame
def generate(survey_question, responses, batch_size, generation_config, relevancy_criteria, ai_gen_criteria):
    model = GenerativeModel("gemini-1.0-pro-001")
    batches = batch_responses(responses, batch_size)
    all_parsed_data = []

    prefix = f"""
    Analyze the given survey responses based on the following criteria:

    Step 1: Relevance Analysis: {relevancy_criteria}

    Step 2: AI Generation Likelihood: {ai_gen_criteria}

    Step 3: Output your analysis in a JSON array with the following structure:
    {{
        "resp": "The verbatim survey response (truncated after 3 words to improve speed and minimize token usage)",
        "rele": "Indicate 'Relevant' if the response adds any value or is related to the question. Mark 'Non-Relevant' only if it is completely devoid of content or context.",
        "rele_r": "Assign a rating from 0 (no relevance) to 10 (high relevance).",
        "ai_gen": "'Low' or 'Medium' or 'High'."
    }}

    IMPORTANT: Analyze all responses provided, regardless of content quality (including duplicates or responses like n/a, none, or nothing). Ensure the number of input responses equals the number of output responses.
    """

    start_time = time.time()

    for i, batch in enumerate(batches):
        prompt = f"""
        {prefix}

        Survey Question: {survey_question}

        Responses:
        {json.dumps(batch)}

        Analyze these responses according to the given criteria. Ensure the output is in JSON format.
        """
        result = model.generate_content(
            prompt,
            generation_config=generation_config
        )
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
        
        time.sleep(6)  # Avoid hitting rate limits

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
    sns.violinplot(x='rele', y='rele_r', data=df, inner='stick')
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
st.set_page_config(layout="wide")
col1, col2 = st.columns([1, 3])
with col1:
    st.image("ppl_logo.jpg", width=100)  # Replace with your image path
with col2:
    st.markdown("<h1 style='text-align: left; color: lightblue;'>Phronesis Apex</h1>", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'project_info'

# Sidebar for settings (only shown after project info is entered)
if st.session_state.step != 'project_info':
    st.sidebar.title("Settings")
    batch_size = st.sidebar.slider('Batch Size', 0, 200 ,50)
    temperature = st.sidebar.slider('Temperature', 0.0, 1.0, 0.2)
    top_k = st.sidebar.number_input('Top K', min_value=1, value=40)
    top_p = st.sidebar.slider('Top P', 0.0, 1.0, 0.8)
    max_output_tokens = st.sidebar.number_input('Max Output Tokens', min_value=1, value=8192)

# Main content area
main_content = st.empty()

# Step 1: Project Info Input
if st.session_state.step == 'project_info':
    with main_content.container():
        st.title("Enter Project Information")
        st.info("Please enter your Google Cloud project name and location to initialize Vertex AI.")
        project_name = st.text_input('Project Name')
        location = st.text_input('Location',"us-central1")
        if st.button('Initialize Vertex AI'):
            if project_name and location:
                try:
                    vertexai.init(project=project_name, location=location)
                    st.session_state.step = 'survey_input'
                    st.rerun()
                except Exception as e:
                    st.error(f"Error initializing Vertex AI: {str(e)}")
            else:
                st.error("Please enter both project name and location.")

# Step 2: Survey Question and Responses Input
elif st.session_state.step == 'survey_input':
    with main_content.container():
        st.title("Enter Survey Question and Responses")
        survey_question = st.text_area('Survey Question')
        responses_text = st.text_area('Survey Responses (one per line)')
        if st.button('Submit Survey Data'):
            if survey_question and responses_text:
                st.session_state.survey_question = survey_question
                st.session_state.responses = responses_text.split('\n')
                st.session_state.step = 'main_app'
                st.rerun()
            else:
                st.error('Please fill in both the survey question and responses.')

# Step 3: Main Application
elif st.session_state.step == 'main_app':
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Submit Open Ends", "AI Analysis", "Visualizations", "Similarity Check", "History"])

    with tab1:
        st.info("Survey question and responses submitted. You can now proceed with analysis or similarity checks in the respective tabs.")
        st.write("Survey Question:", st.session_state.survey_question)
        st.write("Number of responses:", len(st.session_state.responses))

    with tab2:
        st.subheader("Analysis Criteria")
        relevancy_criteria = st.text_area("Relevancy Analysis Criteria", 
            "Classify responses as 'Relevant' unless they add no value or are unrelated to the question. Assess with an inclination towards finding relevance:\n"
            "- Direct Address: Does the response engage with the primary issue, even indirectly? Favor responses addressing any aspect of the question.\n"
            "- Contextual Relevance: Does the response provide insights, context, or perspectives related to the question? Classify as 'Relevant' if it contributes to a broader understanding.\n"
            "- Insightful Contribution: Does the response deepen understanding, provide novel viewpoints, or enrich the discussion? Consider thoughtful or analytically deep responses relevant.\n"
            "- Common Terms: Responses with even one term common with the question must be classified as 'Relevant.'\n"
            "- Open-Ended Responses: Responses like 'none,' 'nothing,' 'not sure,' or 'I am satisfied' should be classified as 'Relevant' as they directly address the question."
        )
        ai_gen_criteria = st.text_area("AI Generation Check Criteria",
            "Evaluate the probability that the response was generated by an AI. Look for indicators like:\n"
            "- Excessive use of adjectives.\n"
            "- Phrasing that is atypical of normal human responses, such as speaking like an eloquent expert.\n"
            "- Overly structured responses.\n"
            "- Absence of grammatical mistakes.\n"
            "- Use of words like 'ignite innovation' and 'delve'."
        )
        if st.button('Analyze Responses'):
            progress_bar = st.progress(0)
            generation_config = GenerationConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
            
            st.session_state.df = generate(st.session_state.survey_question, st.session_state.responses, batch_size, generation_config, relevancy_criteria, ai_gen_criteria)
            st.dataframe(st.session_state.df)
            csv = st.session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "responses_analysis.csv", "text/csv", key='download-csv')

    with tab3:
        if 'df' in st.session_state and st.session_state.df is not None:
            st.subheader("Word Cloud of Responses")
            create_word_cloud(st.session_state.responses)
            st.subheader("Violin Plot of Relevancy Ratings")
            create_violin_plot(st.session_state.df)
        else:
            st.info("Please analyze the responses in the Analysis tab first.")

    with tab4:
        st.subheader("Similarity Check")
        similarity_percent = st.slider("Similarity %", 1, 100, 30)
        threshold = 1 - (similarity_percent / 100)  # Convert percentage to threshold
        if st.button('Run Similarity Check'):
            df_similarity = pd.DataFrame({'response': st.session_state.responses})
            st.session_state.df_clustered = cluster_responses(df_similarity, threshold)
            st.dataframe(st.session_state.df_clustered)

    with tab5:
        st.subheader("History")
        if 'df' in st.session_state and st.session_state.df is not None:
            st.write("Analysis Data:")
            st.dataframe(st.session_state.df)
        if 'df_clustered' in st.session_state and st.session_state.df_clustered is not None:
            st.write("Similarity Data:")
            st.dataframe(st.session_state.df_clustered)
        if not ('df' in st.session_state and st.session_state.df is not None) and not ('df_clustered' in st.session_state and st.session_state.df_clustered is not None):
            st.info("No data available. Please run the analysis or similarity check first.")

if st.sidebar.button('Reset Application'):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
