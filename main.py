import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain.agents import initialize_agent, Tool
from langchain_community.langchain_google_community import GoogleSearchRun
import requests
from PIL import Image
from io import BytesIO

# Check environment variable setup
st.write(
    "Has environment variables been set:",
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
)

# Initialize OpenAI components
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Templates
generate_questions_template = """ you are an expert allergy specialist. 
Analyze the test report which is given as the query/Question.
Given the following Context, generate 10 specific questions asking the patient 
whether they have the corresponding symptoms in the context if the test report/query has the condition \"Positive\". 

Question: {question}
Context: {context}

Answer:"""

analyze_questions_template = """ you are an expert allergy specialist. 
Analyze the Questions and provide an opinion on whether the patient really has those conditions or not.

Questions: {questions}
Answers: {answers}

Expert Opinion:"""

# Add search agent
search_tool = GoogleSearchRun()  # Ensure your API keys for the search service are configured
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
            questions_list = st.session_state["generated_questions"].split("\n")
            images_to_find = ", ".join(questions_list[:5])  # Use first 5 questions for brevity

            # Create a search prompt
            search_prompt = f"Find images of plants or allergens related to: {images_to_find}"
            search_results = agent.run(search_prompt)

            # Extract and display images
            st.subheader("Retrieved Images")
            for result in search_results.split("\n"):
                if result.startswith("http"):  # Check if the result is a URL
                    response = requests.get(result)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        st.image(img, caption=result, use_column_width=True)
        except Exception as e:
            st.error(f"Error during image search: {e}")

# Display opinion if generated
if st.session_state["generated_opinion"]:
    st.subheader("Expert Opinion")
    st.text_area("Generated Opinion:", st.session_state["generated_opinion"], height=300, key="opinion_output")
