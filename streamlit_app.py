import streamlit as st
import numpy as np
from retrieval import retrieve_top_n, get_max_similarity
import google.generativeai as genai
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
    st.success(f"File uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")


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

    # Fetch Gemini API Key from secrets
    gemini_api_key = st.secrets["credentials"]["gemini_api_key"]


    # Configure the Gemini API
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-pro")
        st.success("Gemini API Key successfully configured.")
    except Exception as e:
        st.error(f"An error occurred while setting up the Gemini model: {e}")
        return

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Maddie initiates the conversation
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

    # Display previous chat history
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

    # Initial questions for Maddie
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Select a Question to Begin:")
    initial_questions = [
        "Which of the candidate’s DiSC dimensions is most dominant, and how might that impact their fit for the position?",
        "Which candidate’s communication style aligns most closely with Finance team’s culture?"
    ]
    selected_question = st.selectbox("Choose a question to ask Maddie:", initial_questions, index=0)

    if st.button("Ask Maddie"):
        st.session_state.chat_history.append(("user", selected_question))
        st.chat_message("user").markdown(selected_question)

        # Generate response from Maddie
        try:
            personality_prompt = (
                "Your name is 'Maddie'. You are an HR Analytics specialist in DiSC personality assessment. "
                "You have access to candidate results and can provide insights based on the DiSC framework. "
                "Always be professional, polite, and insightful."
            )

            candidate_results = st.session_state.get("disc_data", "No candidate results found.")
            full_input = f"{personality_prompt}\nCandidate Results: {candidate_results}\nUser: {selected_question}\nAssistant:"

            response = model.generate_content(full_input)
            bot_response = response.text

            # Display bot response
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")

    # Capture free-text user input for further conversation
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Display user input
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        # Generate response from Maddie
        try:
            personality_prompt = (
                "Your name is 'Maddie'. You are an HR Analytics specialist in DiSC personality assessment. "
                "You have access to candidate results and can provide insights based on the DiSC framework. "
                "Always be professional, polite, and insightful."
            )

            candidate_results = st.session_state.get("disc_data", "No candidate results found.")
            full_input = f"{personality_prompt}\nCandidate Results: {candidate_results}\nUser: {user_input}\nAssistant:"

            response = model.generate_content(full_input)
            bot_response = response.text

            # Display bot response
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")


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
    response = st.text_area("Your answer:", key="q1_response", on_change=None)
    
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

    response = st.text_area("Your answer:", key="q2_response")
    
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

    response = st.text_area("Your answer:", key="q3_response")
    
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

    response = st.text_area("Your answer:", key="q4_response")
    
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

    response = st.text_area("Your answer:", key="q5_response")
    
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

    response = st.text_area("Your answer:", key="q6_response")
    
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
    directory = "tmp"
    if not os.path.exists(directory):
         os.makedirs(directory)

    local_file_path = "tmp/candidate_disc_result.csv"
    df.to_csv(local_file_path, index=False)
    st.info(f"CSV saved locally: {os.path.abspath(local_file_path)}")

    # ===== Upload the CSV to GCS =====
    # In the filename, include the name, surname, and timestamp. 
    # For instance: "disc_results/John_Doe_2024-07-31_13-45-00.csv"
    bucket_name = "streamlit-disc-candidate-bucket"
    destination_blob_name = f"disc_results/{candidate_info['name']}_{candidate_info['surname']}_{current_timestamp}.csv"

    upload_to_gcs(bucket_name, destination_blob_name, local_file_path)
    


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

