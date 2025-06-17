from flask import Flask, jsonify, render_template, request
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_groq import ChatGroq
import os
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)

port = int(os.environ.get("PORT", 5000))

# Validate and set API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create FAISS vectorstore
DB_FAISS_PATH = "vectordb"
book_db = None

def load_faiss_database():
    """Load FAISS database with error handling"""
    global book_db
    
    try:
        if os.path.exists(DB_FAISS_PATH):
            print("Loading existing FAISS database...")
            book_db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            print("Successfully loaded existing FAISS database")
        else:
            print("FAISS database not found. Creating empty database...")
            # Create a dummy database - you'll need to populate this with your actual documents
            book_db = FAISS.from_texts(["Medical knowledge base placeholder"], embedding_model)
            print("Created empty FAISS database")
            
    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        print("Creating new database...")
        
        # Remove corrupted database if it exists
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)
            print("Removed corrupted database")
        
        # Create a new empty database
        book_db = FAISS.from_texts(["Medical knowledge base placeholder"], embedding_model)
        print("Created new FAISS database")

# Initialize the database
load_faiss_database()

# Define FAISS retrieval tool
def search_faiss_tool(query: str):
    """Search FAISS database for relevant medical content"""
    try:
        if book_db is None:
            return "Medical knowledge base not available."
        
        docs = book_db.similarity_search(query, k=2)
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        else:
            return "No relevant book content found."
    except Exception as e:
        print(f"Error in FAISS search: {e}")
        return "Error searching medical knowledge base."

# Define web search tool
def search_tavily_tool(query: str):
    """Search web for relevant medical information"""
    try:
        retriever = TavilySearchAPIRetriever(k=3)
        docs = retriever.get_relevant_documents(query)
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        else:
            return "No relevant web content found."
    except Exception as e:
        print(f"Error in web search: {e}")
        return "Error searching web content."

# Define tools for agent
tools = [
    Tool(
        name="MedicalVectorRetriever",
        func=search_faiss_tool,
        description="Use this to search a medical book for context using patient symptoms and vitals."
    ),
    Tool(
        name="WebSearchRetriever",
        func=search_tavily_tool,
        description="Use this when the book doesn't provide enough context; it performs a web search."
    )
]

# Initialize LLM
try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile"
    )
    print("Successfully initialized ChatGroq")
except Exception as e:
    print(f"Error initializing ChatGroq: {e}")
    # Fallback to different model if needed
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant"
    )
    print("Fallback to llama-3.1-8b-instant")

# Initialize Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# Parse agent output
def parse_agent_output(text_data):
    """Parse agent output into structured format"""
    result = {"diseases": [], "tests": [], "tips": []}

    if "POSSIBLE DISEASES:" in text_data and "DIAGNOSTIC TESTS:" in text_data:
        try:
            # Split by sections
            diseases_part = text_data.split("DIAGNOSTIC TESTS:")[0].replace("POSSIBLE DISEASES:", "").strip()
            tests_and_tips_part = text_data.split("DIAGNOSTIC TESTS:")[1]
            
            # Check if TIPS section exists
            if "TIPS:" in tests_and_tips_part:
                tests_part = tests_and_tips_part.split("TIPS:")[0].strip()
                tips_part = tests_and_tips_part.split("TIPS:")[1].strip()
            else:
                tests_part = tests_and_tips_part.strip()
                tips_part = ""

            # Parse diseases
            for line in diseases_part.split('\n'):
                line = line.strip()
                if not line or not line.startswith("-"):
                    continue
                parts = line.lstrip("- ").split(" - ", 1)
                if len(parts) == 2:
                    name, description = parts
                    result["diseases"].append({
                        "name": name.strip(),
                        "description": description.strip()
                    })

            # Parse tests
            for line in tests_part.split('\n'):
                line = line.strip()
                if not line or not line.startswith("-"):
                    continue
                parts = line.lstrip("- ").split(" - ")
                if len(parts) >= 3:
                    name, description, tips = parts[0], parts[1], parts[2]
                elif len(parts) == 2:
                    name, description, tips = parts[0], parts[1], ""
                else:
                    name, description, tips = parts[0], "No description", ""

                result["tests"].append({
                    "name": name.strip(),
                    "description": description.strip(),
                    "tips": tips.strip()
                })

            # Parse overall tips section
            if tips_part:
                for line in tips_part.split('\n'):
                    line = line.strip()
                    if line and line.startswith("-"):
                        tip = line.lstrip("- ").strip()
                        if tip:
                            result["tips"].append(tip)

        except Exception as e:
            print(f"Error parsing agent output: {e}")
            result["tests"].append({
                "name": "Parsing Error",
                "description": "Could not parse agent output properly.",
                "tips": ""
            })
    else:
        result["tests"].append({
            "name": "No diagnostic output",
            "description": "Agent output did not match expected format.",
            "tips": ""
        })
    
    print("Parsed result:", result)
    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def recommend():
    try:
        # Get form data
        symptoms = request.form.get("symptoms", "").strip()
        vitals = request.form.get("vitals", "").strip()
        age = request.form.get("age", "").strip()
        
        # Validate input
        if not symptoms or not age:
            return jsonify({
                "diseases": [],
                "tests": [{
                    "name": "Invalid Input",
                    "description": "Please provide at least age and symptoms.",
                    "tips": ""
                }]
            })
        
        query = f"Patient Age: {age}. Symptoms: {symptoms}. Vitals: {vitals}"
        print(f"Processing query: {query}")
        
        prompt = f"""
You are a medical assistant AI that provides:

1. Possible diseases with one-line descriptions based on the patient's age, symptoms, and vitals.
2. Diagnostic test recommendations with one-line justifications considering the patient's specific profile, and additional suitability tips.
3. Important tips or warnings related to the patient's age, health conditions, pregnancy status, or other relevant factors that may affect the suitability or safety of the recommended tests.

Rules:
- If the input does NOT contain valid medical symptoms, vitals, or age, respond with:
  "I cannot provide diagnostic test recommendations or disease names without valid medical symptoms, vitals, and age."
- For valid inputs, follow this output format exactly:

POSSIBLE DISEASES:
- Disease Name - One line description.
- Disease Name - One line description.

DIAGNOSTIC TESTS:
- Test Name - Short reason - Suitability tips or contraindications(eg. not suitable for children or pregnant women etc.).

Input query:
Patient Age: {age}. Symptoms: {symptoms}. Vitals: {vitals}
"""

        # Run agent
        agent_output = agent.run(prompt)
        print(f"Agent output: {agent_output}")
        
        # Parse output
        parsed_output = parse_agent_output(agent_output)
        
        # Validate parsed output
        if not parsed_output.get("tests", []) and not parsed_output.get("diseases", []):
            return jsonify({
                "diseases": [],
                "tests": [{
                    "name": "No diagnostic output",
                    "description": "Agent could not generate any test recommendations or disease names.",
                    "tips": ""
                }]
            })

        return jsonify(parsed_output)

    except Exception as e:
        print(f"Error in /get endpoint: {str(e)}")
        return jsonify({
            "diseases": [],
            "tests": [{
                "name": "Internal Error",
                "description": f"An error occurred: {str(e)}",
                "tips": ""
            }]
        }), 500

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "faiss_loaded": book_db is not None,
        "groq_configured": GROQ_API_KEY is not None,
        "tavily_configured": TAVILY_API_KEY is not None
    })

if __name__ == "__main__":
    print(f"Starting Flask app on port {port}")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=port)