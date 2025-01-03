import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain.agents import initialize_agent, Tool
from langchain_google_community import GoogleSearchRun, GoogleSearchAPIWrapper
import requests
from PIL import Image
from io import BytesIO

# Load API keys from st.secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["google_api_key"]
os.environ["GOOGLE_CSE_ID"] = "65b214484e5a44069"

# Initialize OpenAI components
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Templates
generate_questions_template = """ you are an expert allergy specialist. 
Analyze the test report which is given as the query/Question.
Given the following Context, generate 10 specific questions asking the patient 
whether they have the corresponding symptoms in the context if the test report/query has the condition "Positive". I dont need headings, summary or numberings. 
format the questions as per the below example:

  [CHAT_STATES.FIRST_VISIT_SYMPTOMS]: "What problem or symptoms do you have?",
  [CHAT_STATES.FIRST_VISIT_DURATION]: "From how long is your symptoms present, please tell me from the first time it started?",
  [CHAT_STATES.FIRST_VISIT_INCREASE]: "Does your symptoms increased in the last few weeks or months if yes please tell me how many months it has increased?",
  [CHAT_STATES.FIRST_VISIT_FREQUENCY]: "Do you have symptoms most of the days? Or few days in a week?",
  [CHAT_STATES.FIRST_VISIT_YEARLY]: "Do you have symptoms throughout the year in all seasons or only few months in a year?",
  [CHAT_STATES.FIRST_VISIT_MONTHS]: "Does your symptoms are severe in certain months of the year?",
  [CHAT_STATES.FIRST_VISIT_TIME]: "At what time of the day or night is your symptoms more?",
  [CHAT_STATES.FIRST_VISIT_OTHER]: "Do you have any other symptoms like cough, wheezing breathing difficulty, Skin allergy, Food Allergy, Drug Allergy",
  [CHAT_STATES.FIRST_VISIT_HEALTH]: "Do you have any other health conditions like High Blood pressure, Diabetes, Heart Issues etc",
  [CHAT_STATES.FIRST_VISIT_FAMILY]: "Does any of your family members have sneezing, wheezing, breathing difficulty, skin allergy, food allergy, drug allergy",
  [CHAT_STATES.FIRST_VISIT_TESTS]: "Have you undergone any allergy test in the past if yes then attach the report",
  [CHAT_STATES.FOLLOWUP_LAST_VISIT]: "When was your last consultation with Dr. Balachandra?",
  [CHAT_STATES.FOLLOWUP_MEDICATION]: "Are you taking the prescribed medications regularly?",
  [CHAT_STATES.FOLLOWUP_IMPROVEMENT]: "How much improvement do you notice in your symptoms? (0-100%)",
  [CHAT_STATES.FOLLOWUP_SYMPTOMS]: "Are you still experiencing any of your previous symptoms?",
  [CHAT_STATES.FOLLOWUP_NEW_SYMPTOMS]: "Have you developed any new symptoms since your last visit?",
  [CHAT_STATES.FOLLOWUP_SIDE_EFFECTS]: "Are you experiencing any side effects from the medications?"

Question: {question}
Context: {context}

Answer:"""

analyze_questions_template = """ you are an expert allergy specialist. 
Analyze the Questions and provide an opinion on whether the patient really has those conditions or not.

Questions: {questions}
Answers: {answers}

Expert Opinion:"""

# Add search agent
api_wrapper = GoogleSearchAPIWrapper(google_api_key=os.environ["GOOGLE_API_KEY"], google_cse_id=os.environ["GOOGLE_CSE_ID"])
search_tool = GoogleSearchRun(api_wrapper=api_wrapper, api_key=os.environ["GOOGLE_API_KEY"])
tools = [Tool(name="Search", func=search_tool.run, description="Search the web for images")]

# Define agent
llm_agent = ChatOpenAI(model_name="gpt-4", temperature=0)
agent = initialize_agent(tools, llm_agent, agent_type="zero-shot-react-description", verbose=True)

st.title("Allergy Opinion Generator")

# Initialize session state for questions and opinion
if "generated_questions" not in st.session_state:
    st.session_state["generated_questions"] = None
if "generated_opinion" not in st.session_state:
    st.session_state["generated_opinion"] = None

st.header("Step 1: Upload Context Documents")

# Upload context documents
context_doc1 = st.file_uploader("Upload the first context document (Docx):", type=["docx"], key="context1")
context_doc2 = st.file_uploader("Upload the second context document (Docx):", type=["docx"], key="context2")

st.header("Step 2: Upload Query Document")
query_doc = st.file_uploader("Upload the query document (Docx):", type=["docx"], key="query")

if st.button("Generate Questions"):
    if context_doc1 and context_doc2 and query_doc:
        # Temporary file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp2, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_query:

            temp1.write(context_doc1.read())
            temp2.write(context_doc2.read())
            temp_query.write(query_doc.read())

            temp1.flush()
            temp2.flush()
            temp_query.flush()

            try:
                # Load and process context documents
                loader1 = Docx2txtLoader(temp1.name)
                loader2 = Docx2txtLoader(temp2.name)
                data1 = loader1.load()
                data2 = loader2.load()

                # Split context into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                docs = text_splitter.split_documents(data1 + data2)

                # Create Vector DB
                db = FAISS.from_documents(docs, embeddings)
                retriever = db.as_retriever()

                # Load query document
                loader_query = Docx2txtLoader(temp_query.name)
                query_data = loader_query.load()

                # Retrieve context and generate questions
                query_context = " ".join([doc.page_content for doc in retriever.invoke(query_data[0].page_content)])
                questions_prompt = generate_questions_template.format(context=query_context, question=query_data[0].page_content)
                questions = llm.invoke(questions_prompt)

                # Save questions to session state
                st.session_state["generated_questions"] = questions.content

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload all three documents.")

# Display questions if generated
if st.session_state["generated_questions"]:
    st.subheader("Generated Questions")
    st.text_area("Generated Questions for the Patient:", st.session_state["generated_questions"], height=300, key="questions_output")

    # Input answers
    st.subheader("Step 3: Provide Answers")
    answers = st.text_area("Enter the patient's answers to the above questions:", key="answers_input")

    if st.button("Generate Opinion"):
        if answers.strip():
            try:
                opinion_prompt = analyze_questions_template.format(questions=st.session_state["generated_questions"], answers=answers)
                opinion = llm.invoke(opinion_prompt)

                # Save opinion to session state
                st.session_state["generated_opinion"] = opinion.content

            except Exception as e:
                st.error(f"Error generating opinion: {e}")
        else:
            st.error("Please provide answers to the generated questions.")

    # Step 4: Search for allergen images
    st.subheader("Step 4: Search for Allergen Images")
    if st.button("Find Relevant Images"):
        try:
            # Ensure questions are available
            if not st.session_state.get("generated_questions"):
                st.error("Please generate questions first.")
                raise Exception("No generated questions found.")

            # Step 1: Generate a search query using LLM
            search_query_prompt = f"""
            You are an expert allergist. Based on the following questions, write concise search queries to find relevant allergen images:
            Questions: {st.session_state['generated_questions']}
            
            Generate separate queries that include the word 'image' to prioritize image results for each allergen or symptom:
            Search Queries:"""

            search_queries = llm.invoke(search_query_prompt).content.strip()

            # Debugging: Display the generated search queries
            st.write("Generated Search Queries:", search_queries)

            if not search_queries:
                st.error("No search queries were generated.")
                


            # Step 2: Perform individual searches for each query
            queries = search_queries.split("\n")  # Each query will be in a new line

            all_retrieved_images = []
            for query in queries:
                query = query.strip()
                if not query:
                    continue  # Skip empty queries

                try:
                    # Perform the search using each query
                    search_results = agent.run(query)

                    # Debugging: Display the search results
                    st.write(f"Search Results for query '{query}':", search_results)

                    # Extract and validate URLs from search results
                    for result in search_results.split("\n"):
                        if result.startswith("http") and (result.endswith(".jpg") or result.endswith(".png")):
                            try:
                                response = requests.get(result, timeout=5)  # Add timeout for robustness
                                if response.status_code == 200 and "image" in response.headers["Content-Type"]:
                                    img = Image.open(BytesIO(response.content))
                                    all_retrieved_images.append((img, result))
                                else:
                                    st.write(f"Skipping non-image URL: {result}")
                            except Exception as e:
                                st.write(f"Error retrieving image from {result}: {e}")
                except Exception as e:
                    st.write(f"Error during image search for query '{query}': {e}")

            # Step 3: Display the images
            st.subheader("Retrieved Images")
            if all_retrieved_images:
                for img, url in all_retrieved_images:
                    st.image(img, caption=url, use_column_width=True)
            else:
                st.write("No valid images retrieved.")

        except Exception as e:
            st.error(f"Error during image search: {e}")

# Display opinion if generated
if st.session_state["generated_opinion"]:
    st.subheader("Expert Opinion")
    st.text_area("Generated Opinion:", st.session_state["generated_opinion"], height=300, key="opinion_output")
