import streamlit as st
import genai

def chat_with_candidate_result_page():
    st.markdown(
        "<h2 style='text-align: center; color: #4CAF50;'>Chat with Maddie Results</h2>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size:16px; color: #4CAF50;'>"
        "👩🏻‍💼 Maddie—your HR companion for gaining valuable insights on a candidate's DiSC profile."
        "</p>",
        unsafe_allow_html=True
    )

    # CSS
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

    # ---------------------
    # 1) Initialize chat history
    # ---------------------
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

    # 2) Display previous chat history
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

    # 3) Some initial questions
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Select a Question to Begin:")
    initial_questions = [
        "Which of the candidate’s DiSC dimensions is most dominant, and how might that impact their fit for the position?",
        "Which candidate’s communication style aligns most closely with Finance team’s culture?"
    ]
    selected_question = st.selectbox("Choose a question to ask Maddie:", initial_questions, index=0)

    if st.button("Ask Maddie"):
        # Add user question to chat history
        st.session_state.chat_history.append(("user", selected_question))
        st.chat_message("user").markdown(selected_question)

        # ---------------------
        # 4) Construct the LLM prompt
        # ---------------------
        try:
            personality_prompt = (
                "Your name is 'Maddie'. You are an HR Analytics specialist in DiSC personality assessment. "
                "You have access to candidate results and can provide insights based on the DiSC framework. "
                "Always be professional, polite, and insightful."
            )
            
            # Load candidate data from session_state 
            candidate_results = st.session_state.get("disc_data", [])
            # Convert that list of dicts into a more readable text
            # For large data, you might want a summary approach instead
            # or a retrieval-based approach. This is a simple example:
            candidate_results_text = "\n".join(
                [str(row_dict) for row_dict in candidate_results]
            )

            # Combine everything into a single prompt
            full_input = (
                f"{personality_prompt}\n\n"
                f"Candidate Results (from CSV):\n{candidate_results_text}\n\n"
                f"User: {selected_question}\n"
                "Assistant:"
            )

            # ---------------------
            # 5) Get response from Gemini
            # ---------------------
            response = model.generate_content(full_input)
            bot_response = response.text

            # Display bot response
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")

    # 6) Capture free-text user input
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

            candidate_results = st.session_state.get("disc_data", [])
            candidate_results_text = "\n".join(
                [str(row_dict) for row_dict in candidate_results]
            )

            full_input = (
                f"{personality_prompt}\n\n"
                f"Candidate Results (from CSV):\n{candidate_results_text}\n\n"
                f"User: {user_input}\n"
                "Assistant:"
            )

            response = model.generate_content(full_input)
            bot_response = response.text

            # Display bot response
            st.session_state.chat_history.append(("assistant", bot_response))
            st.chat_message("assistant").markdown(bot_response)
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
