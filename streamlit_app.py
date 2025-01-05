import streamlit as st
import numpy as np
from retrieval import retrieve_top_n, get_max_similarity
import google.generativeai as palm
from retrieval import vectorizer, chunks
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime
import uuid
import os
import pandas as pd
import io

# initialize GCS credential 
service_account_json = st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
service_account_info = json.loads(service_account_json)
credentials = service_account.Credentials.from_service_account_info(service_account_info)
client = storage.Client(credentials=credentials, project=service_account_info["project_id"])

# def upload df candidaet profile and result to the bucket
def upload_to_gcs(bucket_name, destination_blob_name, local_file_path):
    """
    Uploads a local file to Google Cloud Storage.
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
  #  st.success(f"File uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")


###############################################################################
# 0) Page Navigation
###############################################################################
if "page" not in st.session_state:
    st.session_state.page = "user_selection"

def navigate_to(page_name: str):
    """Helper to switch pages."""
    st.session_state.page = page_name

###############################################################################
# 1) Page: User Selection
###############################################################################
def user_selection_page():
    add_base_styles()
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Welcome to the DiSC Personality Assessment</h1>", unsafe_allow_html=True)
    st.write("Please select whether you are a **Candidate** or **HR**, then press **Submit** to proceed.")

    with st.form("selection_form"):
        user_type = st.radio("Select your role:", ["Candidate", "HR"], index=0)
        submitted = st.form_submit_button("Submit")

        if submitted:
            if user_type == "Candidate":
                navigate_to("candidate_form")
            else:
                navigate_to("hr_form")

###############################################################################
# 2) Page: HR Form (Placeholder)
###############################################################################
# Page: HR Login Form
def hr_form_page():
    st.markdown("<h2 style='text-align: center; color: #FF5722;'>HR Login</h2>", unsafe_allow_html=True)
    st.write("Please log in using your HR credentials.")

    with st.form("hr_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Submit")

        if submitted:
            # Validate using secrets
            if (
                username == st.secrets["credentials"]["username"] and
                password == st.secrets["credentials"]["password"]
            ):
                st.session_state.is_authenticated = True
                navigate_to("chat_with_candidate_result_page")  # Navigate to the chat page
            else:
                st.error("Incorrect username or password!")

###############################################################################
# 3) Page: Chat with Candidate Result
###############################################################################

def add_base_styles():
    """
    Adds base CSS styles to the Streamlit app, such as background color,
    container width, and button styling.
    """
    st.markdown(
        """
        <style>
        body {
            background-color: #f9f9f9;
        }
        /* Container styling */
        .css-1oe6wy4, .css-1y4p8pa {
            max-width: 800px;
            margin: auto;
        }
        /* Button styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

###############################################################################
# 3) Maddie Chatbot
###############################################################################

import json
import io

def merge_csv_from_gcs(bucket_name, prefix):
    """Helper function to merge multiple CSVs from GCS into a single DataFrame."""
    # --- Initialize GCS credential ---
    service_account_json = st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    service_account_info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    storage_client = storage.Client(credentials=credentials, project=service_account_info["project_id"])
    
    # Get the bucket and list CSV blobs
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    df_list = []
    for blob in blobs:
        if blob.name.endswith(".csv"):
            csv_data = blob.download_as_text()
            df_temp = pd.read_csv(io.StringIO(csv_data))
            df_list.append(df_temp)
    
    # Merge all DataFrames
    if df_list:
        df_merged = pd.concat(df_list, ignore_index=True)
        return df_merged
    else:
        return pd.DataFrame()

def get_position_trait_data(bucket_name, file_path):
    """Fetch the position_trait_v1.csv from GCS and return as a DataFrame."""
    # --- Initialize GCS credential ---
    service_account_json = st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    service_account_info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    storage_client = storage.Client(credentials=credentials, project=service_account_info["project_id"])

    # Fetch the CSV
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    csv_data = blob.download_as_text()
    df_position_trait = pd.read_csv(io.StringIO(csv_data))
    return df_position_trait

def chat_with_candidate_result_page():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Chat with Maddie Results</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size:16px; color: #4CAF50;'>"
        "👩🏻‍💼 Maddie—your HR companion for gaining valuable insights on a candidate's DiSC profile."
        "</p>",
        unsafe_allow_html=True
    )

    # Custom CSS for larger emoji
    st.markdown("""
        <style>
        .chat-emoji {
            font-size: 50px;  /* Adjust size as needed */
            margin-right: 10px;
        }
        .chat-maddie {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .chat-maddie .message {
            background-color: #FFF4E0;
            border-radius: 10px;
            padding: 10px 15px;
            margin-left: 10px;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 1) CONFIGURE Google Generative AI (Palm) with your key
    gemini_api_key = st.secrets["credentials"]["gemini_api_key"]
    try:
        palm.configure(api_key=gemini_api_key)
        st.success("Gemini (Palm) API Key successfully configured.")
    except Exception as e:
        st.error(f"An error occurred while setting up the Gemini model: {e}")
        return

    # 2) PREPARE DATA
    bucket_name = "streamlit-disc-candidate-bucket"

    # Merge all CSVs within disc_results/
    prefix = "disc_results/"
    df_merged = merge_csv_from_gcs(bucket_name, prefix)

    # Load the position_trait_v1.csv
    trait_file_path = "position_trait/position_trait_v1.csv"
    df_position_trait = get_position_trait_data(bucket_name, trait_file_path)

    # Store data in session state for re-use
    if "df_merged" not in st.session_state:
        st.session_state["df_merged"] = df_merged
    if "df_position_trait" not in st.session_state:
        st.session_state["df_position_trait"] = df_position_trait

    # 3) INITIALIZE CHAT HISTORY
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        initial_message = (
            "Hello, I’m Maddie—your dedicated HR companion for DiSC insights! "
            "I’m here to help you evaluate how each candidate’s DiSC profile aligns with the role you’re trying to fill. "
            "Ready to dive in?"
        )
        st.session_state.chat_history.append(("assistant", initial_message))
        st.markdown(
            f"""
            <div class="chat-maddie">
                <div class="chat-emoji">👩🏻‍💼</div>
                <div class="message">{initial_message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4) DISPLAY EXISTING CHAT
    for role, message in st.session_state.chat_history:
        if role == "assistant":
            st.markdown(
                f"""
                <div class="chat-maddie">
                    <div class="chat-emoji">👩🏻‍💼</div>
                    <div class="message">{message}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.chat_message(role).markdown(message)

    # 5) PRESENT INITIAL QUESTION CHOICES (No "Ask Maddie" button—just the selectbox)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Select a Question to Begin:")
    initial_questions = [
        "Would you like me to help you understand candidate DiSC personality better for hiring?",
        "Guiding DiSC profile for new position"
    ]
    selected_question = st.selectbox("Choose topic to ask Maddie:", initial_questions, index=0)

    def generate_text_from_google(prompt_text: str) -> str:
        """Call Google's generative AI and return the text result."""
        model_name = "models/text-bison-001"  # Or "models/chat-bison-001"
        try:
            response = palm.generate_text(
                model=model_name,
                prompt=prompt_text,
                temperature=0.7,
                candidate_count=1
            )
            return response.result
        except Exception as e:
            st.error(f"Error calling Palm API: {e}")
            return "Sorry, I had an issue generating a response."

    # If the user chooses the first question:
    if selected_question == "Would you like me to help you understand candidate DiSC personality better for hiring?":
        # We display the positions and the confirm button
        positions = [
            "Project Manager", "Product Owner", "HR Manager", "Accounting Manager",
            "Finance Manager", "Sale Manager", "Operation Manager", "Data Analyst",
            "Data Engineer", "Data Science", "Marketing Manager", "Strategy Manager",
            "Procurement Manager"
        ]
        st.markdown("**Please select the position that you would like to hire:**")
        selected_position = st.radio("Positions:", positions, key="selected_position_radio")

        if st.button("Confirm Position"):
            # Append user message
            user_msg = f"I want to hire a {selected_position}."
            st.session_state.chat_history.append(("user", user_msg))
            st.chat_message("user").markdown(user_msg)
            
            # Do the logic to find top 5
            df_position_trait = st.session_state["df_position_trait"]
            df_merged = st.session_state["df_merged"]

            row = df_position_trait[df_position_trait["position"] == selected_position]
            if row.empty:
                msg = f"Oops! I don't have trait data for the position: {selected_position}"
                st.session_state.chat_history.append(("assistant", msg))
                st.chat_message("assistant").markdown(msg)
            else:
                primary_disc = row.iloc[0]["primary_disc"]
                secondary_disc = row.iloc[0]["secondary_disc"]
                reason = row.iloc[0]["reason"]

                disc_col_map = {
                    "D": "D score percentage",
                    "I": "I score percentage",
                    "S": "S score percentage",
                    "C": "C score percentage"
                }
                primary_col = disc_col_map.get(primary_disc, None)
                secondary_col = disc_col_map.get(secondary_disc, None)

                if primary_col is None or secondary_col is None:
                    msg = "Unable to map DiSC trait columns. Please check your data."
                    st.session_state.chat_history.append(("assistant", msg))
                    st.chat_message("assistant").markdown(msg)
                else:
                    df_merged["score_sum"] = df_merged[primary_col] + df_merged[secondary_col]
                    df_sorted = df_merged.sort_values(by="score_sum", ascending=False)
                    top_5 = df_sorted.head(5)

                    # Summarize top 5
                    candidate_list = []
                    for idx, row_ in top_5.iterrows():
                        candidate_list.append(
                            f"- **Name**: {row_['Name']} {row_['Surname']} | "
                            f"Primary DiSC: {row_['DiSC Result']} | "
                            f"{primary_disc}={row_[primary_col]:.2f} & {secondary_disc}={row_[secondary_col]:.2f}"
                        )
                    candidates_text = "\n".join(candidate_list)

                    msg = (
                        f"**Position**: {selected_position}\n\n"
                        f"**Ideal DiSC**: Primary={primary_disc}, Secondary={secondary_disc}\n\n"
                        f"**Reason**: {reason}\n\n"
                        f"**Top 5 candidates** based on `{primary_col}` + `{secondary_col}`:\n"
                        f"{candidates_text}"
                    )
                    st.session_state.chat_history.append(("assistant", msg))
                    st.chat_message("assistant").markdown(msg)

    else:
        # If user chooses "Guiding DiSC profile for new position" or anything else
        # We can automatically show the LLM response or you can handle it differently
        personality_prompt = (
            "Your name is 'Maddie'. You are an HR Analytics specialist in DiSC personality assessment. "
            "You have access to candidate results and can provide insights based on the DiSC framework. "
            "Always be professional, polite, and insightful."
        )
        # We'll add a small text to clarify
        user_msg = f"User selected: {selected_question}"
        st.session_state.chat_history.append(("user", user_msg))
        st.chat_message("user").markdown(user_msg)

        bot_response = generate_text_from_google(f"{personality_prompt}\nUser: {selected_question}\nAssistant:")
        st.session_state.chat_history.append(("assistant", bot_response))
        st.chat_message("assistant").markdown(bot_response)

    # 6) FREE-TEXT USER INPUT
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        personality_prompt = (
            "Your name is 'Maddie'. You are an HR Analytics specialist in DiSC personality assessment. "
            "You have access to candidate results and can provide insights based on the DiSC framework. "
            "Always be professional, polite, and insightful."
        )
        full_input = f"{personality_prompt}\nUser: {user_input}\nAssistant:"
        bot_response = generate_text_from_google(full_input)
        st.session_state.chat_history.append(("assistant", bot_response))
        st.chat_message("assistant").markdown(bot_response)


###############################################################################
# 3) Page: Candidate Form
###############################################################################
def candidate_form_page():
    add_base_styles()
    st.markdown("<h2 style='text-align: center; color: #2196F3;'>Candidate Information</h2>", unsafe_allow_html=True)
    st.write("Please provide your details below.")

    with st.form("candidate_form"):
        name = st.text_input("Name")
        surname = st.text_input("Surname")
        age = st.number_input("Age", min_value=18, max_value=99, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        applied_position = st.selectbox("Applied Position", ["Project Manager", "Product Owner", "HR Manager", "Accounting Manager", "Finance Manager", "Sale Manager", "Operation Manager", "Data Analyst", "Data Engineer", "Data Science", "Marketing Manager", "Strategy Manager", "Procurement Manager"])
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not all([name, surname, age, gender, applied_position]):
                st.error("Please fill out all fields!")
            else:
                st.session_state.candidate_data = {
                    "name": name,
                    "surname": surname,
                    "age": age,
                    "gender": gender,
                    "applied_position": applied_position
                }

                # We'll store final disc data (similarities) here
                if "disc_data" not in st.session_state:
                    st.session_state.disc_data = {}
                
                navigate_to("question_1")

###############################################################################
# 4) QUESTION PAGES (Q1..Q6)
#    - Minimum 15 words (real-time display: N/15)
#    - Similarity threshold = 0.4
#    - Store each question’s D/I/S/C in disc_data[Qn]
###############################################################################

def question_1_page():
    add_base_styles()
    st.title("Question 1")
    st.write("How do you handle a situation where you need to change your plan — adapt quickly or proceed carefully?")

    #st.markdown(
       # """
        #<div style="
            #background-color: #FAFAFA;
            #border-left: 4px solid #007ACC;
            #padding: 10px;
            #margin-top: 10px;
            #border-radius: 4px;
       # ">
            #<p style="font-size:15px; color:#333; margin:0;">
               # <strong>More detail:</strong><br>
                #&nbsp;&nbsp;- Do you adapt quickly and make adjustments on the fly without much hesitation? or<br>
                #&nbsp;&nbsp;- Do you prefer to pause, assess the situation, and carefully consider how to proceed before making changes?
          #  </p>
       #</div>
       # """,
       # unsafe_allow_html=True
    #)

    # Real-time text area with a session key
    response = st.text_area("Your answer - please answer in English:", key="q1_response", on_change=None)
    
    # Real-time word count
    word_count = len(st.session_state.q1_response.strip().split())
    st.info(f"Words typed: {word_count}/15")

    if st.button("Submit Answer"):
        # Final check once they press Submit
        if word_count < 15:
            st.warning("The answer is too short, please provide more detail.")
            return

        sim = get_max_similarity(st.session_state.q1_response, "Q1")
        if sim < 0.4:
            st.warning("Your answer is not relevant. Please answer again.")
            return

        sims = retrieve_top_n(st.session_state.q1_response, "Q1", n=4)
        st.session_state.disc_data["Q1"] = {
            "answer": st.session_state.q1_response,
            "word_count": word_count,
            "similarities": sims
        }
        navigate_to("question_2")


def question_2_page():
    add_base_styles()
    st.title("Question 2")
    st.write("How do you approach deadlines and managing your time — get things done quickly or plan thoughtfully?")

    #st.markdown(
        #"""
        #<div style="
            #background-color: #FAFAFA;
            #border-left: 4px solid #007ACC;
            #padding: 10px;
            #margin-top: 10px;
            #border-radius: 4px;
        #">
            #<p style="font-size:15px; color:#333; margin:0;">
               # <strong>More detail:</strong><br>
               # &nbsp;&nbsp;- Do you respond to time constraints immediately, focusing on getting things done as quickly as possible, even under pressure? or<br>
               # &nbsp;&nbsp;- Do you take a step back, allocate time carefully, and plan things out before taking action?
          #  </p>
       # </div>
       # """,
       # unsafe_allow_html=True
    #)

    response = st.text_area("Your answer - please answer in English:", key="q2_response")
    
    word_count = len(st.session_state.q2_response.strip().split())
    st.info(f"Words typed: {word_count}/15")

    if st.button("Submit Answer"):
        if word_count < 15:
            st.warning("The answer is too short, please provide more detail.")
            return

        sim = get_max_similarity(st.session_state.q2_response, "Q2")
        if sim < 0.4:
            st.warning("Your answer is not relevant. Please answer again.")
            return

        sims = retrieve_top_n(st.session_state.q2_response, "Q2", n=4)
        st.session_state.disc_data["Q2"] = {
            "answer": st.session_state.q2_response,
            "word_count": word_count,
            "similarities": sims
        }
        navigate_to("question_3")


def question_3_page():
    add_base_styles()
    st.title("Question 3")
    st.write("When faced with conflict, do you prefer to resolve it directly through clear discussion, or focus on maintaining harmony and preserving relationships?")

    #st.markdown(
       #"""
        #<div style="
            #background-color: #FAFAFA;
            #border-left: 4px solid #007ACC;
            #padding: 10px;
            #margin-top: 10px;
            #border-radius: 4px;
       # ">
            #<p style="font-size:15px; color:#333; margin:0;">
                #<strong>More detail:</strong><br>
                #&nbsp;&nbsp;- Do you address conflict directly, questioning the issue and seeking to resolve it through clear discussion? or<br>
                #&nbsp;&nbsp;- Do you focus on finding common ground, maintaining harmony, and avoiding unnecessary tension?
           # </p>
       # </div>
        #""",
       # unsafe_allow_html=True
   # )

    response = st.text_area("Your answer - please answer in English:", key="q3_response")
    
    word_count = len(st.session_state.q3_response.strip().split())
    st.info(f"Words typed: {word_count}/15")

    if st.button("Submit Answer"):
        if word_count < 15:
            st.warning("The answer is too short, please provide more detail.")
            return

        sim = get_max_similarity(st.session_state.q3_response, "Q3")
        if sim < 0.4:
            st.warning("Your answer is not relevant. Please answer again.")
            return

        sims = retrieve_top_n(st.session_state.q3_response, "Q3", n=4)
        st.session_state.disc_data["Q3"] = {
            "answer": st.session_state.q3_response,
            "word_count": word_count,
            "similarities": sims
        }
        navigate_to("question_4")

def question_4_page():
    add_base_styles()
    st.title("Question 4")
    st.write("When presented with a new idea, do you consider its impact on people, question its validity, or welcome it with optimism?")

    #st.markdown(
       # """
        #<div style="
          #  background-color: #FAFAFA;
           # border-left: 4px solid #007ACC;
           # padding: 10px;
           # margin-top: 10px;
           # border-radius: 4px;
       # ">
            #<p style="font-size:15px; color:#333; margin:0;">
               # <strong>More detail:</strong><br>
               # &nbsp;&nbsp;- Do you approach it with skepticism, asking questions to understand its validity and implications? or<br>
               # &nbsp;&nbsp;- Are you open and accepting, embracing the idea with enthusiasm and trust in its potential?
          #  </p>
       # </div>
      #  """,
      #  unsafe_allow_html=True
    #)

    response = st.text_area("Your answer - please answer in English:", key="q4_response")
    
    word_count = len(st.session_state.q4_response.strip().split())
    st.info(f"Words typed: {word_count}/15")

    if st.button("Submit Answer"):
        if word_count < 15:
            st.warning("The answer is too short, please provide more detail.")
            return

        sim = get_max_similarity(st.session_state.q4_response, "Q4")
        if sim < 0.4:
            st.warning("Your answer is not relevant. Please answer again.")
            return

        sims = retrieve_top_n(st.session_state.q4_response, "Q4", n=4)
        st.session_state.disc_data["Q4"] = {
            "answer": st.session_state.q4_response,
            "word_count": word_count,
            "similarities": sims
        }
        navigate_to("question_5")

def question_5_page():
    add_base_styles()
    st.title("Question 5")
    st.write("When you have a new task, do you jump in quickly to get started, or take time to understand it fully, gather information, and create a plan?")

    #st.markdown(
       # """
      #  <div style="
        #    background-color: #FAFAFA;
        #    border-left: 4px solid #007ACC;
        #    padding: 10px;
        #    margin-top: 10px;
        #    border-radius: 4px;
       # ">
           # <p style="font-size:15px; color:#333; margin:0;">
           #     <strong>More detail:</strong><br>
           #     &nbsp;&nbsp;- Do you jump in quickly, eager to get started without much delay? or<br>
           #     &nbsp;&nbsp;- Do you prefer to take your time to fully understand the task, gather information, and create a plan before starting?
          #  </p>
       # </div>
       # """,
      #  unsafe_allow_html=True
     #)

    response = st.text_area("Your answer - please answer in English:", key="q5_response")
    
    word_count = len(st.session_state.q5_response.strip().split())
    st.info(f"Words typed: {word_count}/15")

    if st.button("Submit Answer"):
        if word_count < 15:
            st.warning("The answer is too short, please provide more detail.")
            return

        sim = get_max_similarity(st.session_state.q5_response, "Q5")
        if sim < 0.4:
            st.warning("Your answer is not relevant. Please answer again.")
            return

        sims = retrieve_top_n(st.session_state.q5_response, "Q5", n=4)
        st.session_state.disc_data["Q5"] = {
            "answer": st.session_state.q5_response,
            "word_count": word_count,
            "similarities": sims
        }
        navigate_to("question_6")

def question_6_page():
    add_base_styles()
    st.title("Question 6")
    st.write("When making a decision, do you focus on goals, rely on your judgment, or consider others' perspectives?")

    #st.markdown(
       # """
        #<div style="
           # background-color: #FAFAFA;
           # border-left: 4px solid #007ACC;
           # padding: 10px;
           # margin-top: 10px;
           # border-radius: 4px;
      #  ">
           # <p style="font-size:15px; color:#333; margin:0;">
              #  <strong>More detail:</strong><br>
              #  &nbsp;&nbsp;- Do you rely on your own judgment, ask critical questions, and carefully evaluate all options before deciding? or<br>
              #  &nbsp;&nbsp;- Do you consider other people perspectives, and collaborate in the decision-making process?
           # </p>
      #  </div>
      #  """,
      #  unsafe_allow_html=True
    #)

    response = st.text_area("Your answer - please answer in English:", key="q6_response")
    
    word_count = len(st.session_state.q6_response.strip().split())
    st.info(f"Words typed: {word_count}/15")

    if st.button("Submit Answer"):
        if word_count < 15:
            st.warning("The answer is too short, please provide more detail.")
            return

        sim = get_max_similarity(st.session_state.q6_response, "Q6")
        if sim < 0.4:
            st.warning("Your answer is not relevant. Please answer again.")
            return

        sims = retrieve_top_n(st.session_state.q6_response, "Q6", n=4)
        st.session_state.disc_data["Q6"] = {
            "answer": st.session_state.q6_response,
            "word_count": word_count,
            "similarities": sims
        }
        navigate_to("disc_result")

###############################################################################
# 5) DISC RESULT PAGE: final type + description
###############################################################################
def disc_result_page():
    add_base_styles()
    st.title("Your DiSC Assessment Results")

    if "disc_data" not in st.session_state or len(st.session_state.disc_data) < 6:
        st.warning("Please complete all 6 questions first.")
        return

    disc_data = st.session_state.disc_data

    # Summation across all Qs
    total_D = 0.0
    total_I = 0.0
    total_S = 0.0
    total_C = 0.0

    # เราทราบว่า st.session_state.disc_data[qid]["similarities"]
    # คือ list ของ (similarity, chunk)
    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        sims = st.session_state.disc_data[qid]["similarities"]
        for sim, chunk in sims:
            disc_type = chunk["type"]  # ตรงกับใน retrieval.py
            if disc_type == "D":
                total_D += sim
            elif disc_type == "I":
                total_I += sim
            elif disc_type == "S":
                total_S += sim
            elif disc_type == "C":
                total_C += sim

    # Normalize to percentages
    total_all = total_D + total_I + total_S + total_C
    if total_all <= 0:
        st.error("No valid similarity data found. Please ensure your answers were relevant.")
        return

    pct_D = (total_D / total_all) * 100
    pct_I = (total_I / total_all) * 100
    pct_S = (total_S / total_all) * 100
    pct_C = (total_C / total_all) * 100


    # Pick the best dimension
    final_scores = {"D": total_D, "I": total_I, "S": total_S, "C": total_C}
    best_type = max(final_scores, key=final_scores.get)
    
    # --- Generate timestamp and unique ID ---
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # e.g., "2024-07-31_13-45-00"
    unique_id = str(uuid.uuid4())  # e.g., "bd65600d-8669-4903-8a14-af88203add38"

    # Gather candidate info
    candidate_info = st.session_state["candidate_data"]  # from candidate_form_page

    # Create DataFrame
    df = pd.DataFrame([{
        "Unique ID": unique_id,
        "Timestamp": current_timestamp,           # store timestamp
        "Name": candidate_info["name"],
        "Surname": candidate_info["surname"],
        "Age": candidate_info["age"],
        "Gender": candidate_info["gender"],
        "Applied Position": candidate_info["applied_position"],
        "DiSC Result": best_type,
        "D score percentage": pct_D,
        "I score percentage": pct_I,
        "S score percentage": pct_S,
        "C score percentage": pct_C
    }])

    # --- LAYOUT: left for text, right for graph
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Your DiSC Distribution:")
        st.write(f"- **D (Dominance)**: {pct_D:.2f}%")
        st.write(f"- **I (Influence)**: {pct_I:.2f}%")
        st.write(f"- **S (Steadiness)**: {pct_S:.2f}%")
        st.write(f"- **C (Conscientiousness)**: {pct_C:.2f}%")
        st.success(f"Your primary DiSC style seems to be: {best_type}")


    # --- Build the quadrant graph on the right
    with col_right:
        fig, ax = plt.subplots(figsize=(4, 4))

        # Draw the four color “squares” 
        # D=green (top-left), I=red (top-right), C=yellow (bottom-left), S=blue (bottom-right)
        ax.add_patch(patches.Rectangle((-100, 0), 100, 100, 
                                       facecolor="#A8D5A2", alpha=0.3))  # top-left D
        ax.add_patch(patches.Rectangle((0, 0), 100, 100, 
                                       facecolor="#F4A8A8", alpha=0.3))  # top-right I
        ax.add_patch(patches.Rectangle((-100, -100), 100, 100, 
                                       facecolor="#FAF3AD", alpha=0.3)) # bottom-left C
        ax.add_patch(patches.Rectangle((0, -100), 100, 100, 
                                       facecolor="#A8D7F4", alpha=0.3))  # bottom-right S

        # Thin lines for X=0 and Y=0
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)

        # Place transparent big letters in each quadrant
        ax.text(-50, 50, "D", fontsize=100, color="gray", alpha=0.2,
                ha="center", va="center")
        ax.text(50, 50, "I", fontsize=100, color="gray", alpha=0.2,
                ha="center", va="center")
        ax.text(-50, -50, "C", fontsize=100, color="gray", alpha=0.2,
                ha="center", va="center")
        ax.text(50, -50, "S", fontsize=100, color="gray", alpha=0.2,
                ha="center", va="center")

        # Coordinates by your quadrant rule:
        # I => (+pct_I, +pct_I)  [top-right]
        # D => (-pct_D, +pct_D)  [top-left]
        # C => (-pct_C, -pct_C)  [bottom-left]
        # S => (+pct_S, -pct_S)  [bottom-right]
        xD, yD = -pct_D, +pct_D
        xI, yI = +pct_I, +pct_I
        xC, yC = -pct_C, -pct_C
        xS, yS = +pct_S, -pct_S

        # Plot each point
        ax.plot(xD, yD, 'ko', markersize=6)
        ax.plot(xI, yI, 'ko', markersize=6)
        ax.plot(xC, yC, 'ko', markersize=6)
        ax.plot(xS, yS, 'ko', markersize=6)

        # Connect them with a polygon, e.g. D -> I -> S -> C -> D
        ax.plot([xD, xI, xS, xC, xD],
                [yD, yI, yS, yC, yD],
                color='black', linewidth=1)

        # Set the bounds & remove ticks
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        st.pyplot(fig)

    # Show the standard description
    show_disc_description(best_type)
    
    st.success(f"Thank you very much for your time. Please go through the link to evaluate this project : https://forms.gle/oPcrYYaDc1FwhuA26 ")

    # ===== Save the CSV locally =====
    # You can choose any file path. We'll just keep it in the current working directory for clarity.
    #local_file_path = "tmp/candidate_disc_result.csv"
    #df.to_csv(local_file_path, index=False)
    #st.info(f"CSV saved locally: {os.path.abspath(local_file_path)}")

    # ===== Save the CSV locally =====
    # Ensure the tmp directory exists

    base_dir = os.path.dirname(os.path.abspath(__file__))  # ตำแหน่งเดียวกับไฟล์ streamlit_app.py
    tmp_dir = os.path.join(base_dir, "tmp")

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    local_file_path = os.path.join(tmp_dir, "candidate_disc_result.csv")
    df.to_csv(local_file_path, index=False)
    #st.info(f"CSV saved locally: {os.path.abspath(local_file_path)}")

    # ===== Upload the CSV to GCS =====
    # In the filename, include the name, surname, and timestamp. 
    # For instance: "disc_results/John_Doe_2024-07-31_13-45-00.csv"
    bucket_name = "streamlit-disc-candidate-bucket"
    destination_blob_name = f"disc_results/{candidate_info['name']}_{candidate_info['surname']}_{current_timestamp}.csv"

    upload_to_gcs(bucket_name, destination_blob_name, local_file_path)
    
    # --- Save detailed answers + similarities as a new CSV ---
    # Build a row for each question: answer, D similarity, I similarity, S similarity, C similarity
    rows = []
    for qid in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        answer_text = st.session_state.disc_data[qid]["answer"]
        # Initialize D, I, S, C for each question
        similarity_map = {"D": 0.0, "I": 0.0, "S": 0.0, "C": 0.0}
        for sim, chunk in st.session_state.disc_data[qid]["similarities"]:
            disc_type = chunk["type"]
            # Assign that similarity
            similarity_map[disc_type] = sim

        row = {
            "answer": answer_text,
            "D cosine similarity": similarity_map["D"],
            "I cosine similarity": similarity_map["I"],
            "S cosine similarity": similarity_map["S"],
            "C cosine similarity": similarity_map["C"]
        }
        rows.append(row)

    df_answers = pd.DataFrame(
        rows, 
        columns=[
            "answer",
            "D cosine similarity",
            "I cosine similarity",
            "S cosine similarity",
            "C cosine similarity",
        ]
    )

    # Save the new answers DataFrame as CSV locally
    answers_file_path = os.path.join(tmp_dir, "candidate_answers_disc_sim.csv")
    df_answers.to_csv(answers_file_path, index=False)
    #st.info(f"Answers + Similarities CSV saved locally: {os.path.abspath(answers_file_path)}")

    # Upload this CSV to a *new folder* in GCS
    # The user wants something like: "(New folder name)/Name_Surname_2025-01-02_12-34-56.csv"
    new_folder_blob = f"(answers_result)/{candidate_info['name']}_{candidate_info['surname']}_{current_timestamp}.csv"
    upload_to_gcs(bucket_name, new_folder_blob, answers_file_path)
   # st.success("Detailed answers + similarity file has been uploaded to GCS in the new folder!")


def show_disc_description(disc_type):
    """Display the final text block for the chosen type."""
    if disc_type == "D":
        st.markdown("""
**You are Dominance!**

As a Dominance personality, you are bold, confident, and results-driven. You thrive on challenges and excel at making quick decisions. Your assertiveness and focus on goals make you a natural leader.

- **Strengths**: Leadership, decisiveness, and driving results.
- **Challenges**: Can be too direct or impatient, sometimes overlooking others’ input.
- **Best Environment**: Fast-paced, competitive workplaces with clear goals.
- **Tips for Growth**: Practice active listening and balance your task focus with empathy.
""")
    elif disc_type == "I":
        st.markdown("""
**You are Influence!**

As an Influence personality, you are charismatic, enthusiastic, and a connector. You energize others and thrive in social settings, building relationships with ease.

- **Strengths**: Networking, positivity, and inspiring collaboration.
- **Challenges**: May struggle with focus or follow-through.
- **Best Environment**: Dynamic, people-centered workplaces with room for creativity.
- **Tips for Growth**: Work on time management and prioritize tasks to channel your enthusiasm effectively.
""")
    elif disc_type == "S":
        st.markdown("""
**You are Steadiness!**

As a Steadiness personality, you are dependable, empathetic, and calm. You prioritize harmony and are a reliable, supportive presence in any team.

- **Strengths**: Patience, reliability, and fostering collaboration.
- **Challenges**: Avoids conflict and resists change.
- **Best Environment**: Stable, supportive workplaces emphasizing trust.
- **Tips for Growth**: Embrace change and assert your needs confidently.
""")
    elif disc_type == "C":
        st.markdown("""
**You are Conscientiousness!**

As a Conscientiousness personality, you are analytical, detail-oriented, and committed to excellence. You prefer structure and thrive on accuracy.

- **Strengths**: Precision, problem-solving, and high standards.
- **Challenges**: Can be overly critical or perfectionistic.
- **Best Environment**: Structured, detail-focused settings valuing expertise.
- **Tips for Growth**: Focus on progress over perfection and trust others’ abilities.
""")

###############################################################################
# (OPTIONAL) STYLES
###############################################################################
def add_base_styles():
    st.markdown("""
        <style>
        body {
            background-color: #f9f9f9;
        }
        .css-1oe6wy4, .css-1y4p8pa {  
            max-width: 800px;
            margin: auto;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


###############################################################################
# MAIN: Router
###############################################################################
def main():
    if "page" not in st.session_state:
        st.session_state.page = "user_selection"

    # Router logic
    page = st.session_state.page
    if page == "user_selection":
        user_selection_page()
    elif page == "hr_form":
        hr_form_page()
    elif page == "chat_with_candidate_result_page":
        chat_with_candidate_result_page()
    elif page == "candidate_form":
        candidate_form_page()
    elif page == "question_1":
        question_1_page()
    elif page == "question_2":
        question_2_page()
    elif page == "question_3":
        question_3_page()
    elif page == "question_4":
        question_4_page()
    elif page == "question_5":
        question_5_page()
    elif page == "question_6":
        question_6_page()
    elif page == "disc_result":
        disc_result_page()
    else:
        user_selection_page()

if __name__ == "__main__":
    main()

