from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_groq import ChatGroq
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Get port from environment (Elastic Beanstalk uses PORT, but we'll also check for common alternatives)
port = int(os.environ.get("PORT", os.environ.get("FLASK_PORT", 5000)))

# Validate required environment variables
required_env_vars = ["TAVILY_API_KEY", "GROQ_API_KEY", "mongo_uri"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

# Set environment variables
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("mongo_uri")

# Initialize MongoDB connection with error handling
try:
    client = MongoClient(mongo_uri)
    # Test connection
    client.admin.command('ping')
    db = client["vector"]
    collection = db["vectorembeddings"]
    logger.info("Successfully connected to MongoDB Atlas")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Initialize embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise

# Initialize MongoDB Atlas vector store (NO FAISS)
try:
    book_db = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name="vector_index2",
        connection_kwargs={"tls": True}
    )
    logger.info("MongoDB Atlas vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB Atlas vector store: {e}")
    raise

# Define MongoDB retrieval tool
def search_mongo_tool(query: str):
    """Search MongoDB Atlas vector store for medical information"""
    try:
        logger.info(f"Searching MongoDB for: {query}")
        docs = book_db.similarity_search(query, k=3)
        if docs:
            result = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Found {len(docs)} relevant documents in MongoDB")
            return result
        else:
            logger.warning("No relevant documents found in MongoDB")
            return "No relevant medical book content found."
    except Exception as e:
        logger.error(f"MongoDB search error: {e}")
        return "Error searching medical database."

# Define Tavily web search tool
def search_tavily_tool(query: str):
    """Search web using Tavily API for additional medical information"""
    try:
        logger.info(f"Searching web for: {query}")
        retriever = TavilySearchAPIRetriever(k=5)
        docs = retriever.get_relevant_documents(query)
        if docs:
            result = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Found {len(docs)} relevant web documents")
            return result
        else:
            logger.warning("No relevant web content found")
            return "No relevant web content found."
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return "Error searching web content."

# Define tools for the agent
tools = [
    Tool(
        name="MedicalVectorRetriever",
        func=search_mongo_tool,
        description="Search the medical knowledge base using patient symptoms and vitals to find relevant medical information, disease patterns, and diagnostic insights."
    ),
    Tool(
        name="WebSearchRetriever", 
        func=search_tavily_tool,
        description="Search the web for current medical information, recent studies, and additional diagnostic guidance when the medical knowledge base lacks sufficient information."
    )
]

# Initialize Groq LLM
try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.1  # Low temperature for more consistent medical responses
    )
    logger.info("Groq LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {e}")
    raise

# Initialize LangChain agent
try:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Limit iterations to prevent infinite loops
        early_stopping_method="generate"
    )
    logger.info("LangChain agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    raise

def parse_agent_output(text_data):
    """Parse agent output into structured JSON format"""
    result = {"diseases": [], "tests": []}
    
    try:
        if "POSSIBLE DISEASES:" in text_data and "DIAGNOSTIC TESTS:" in text_data:
            # Split the text into sections
            diseases_part = text_data.split("DIAGNOSTIC TESTS:")[0].replace("POSSIBLE DISEASES:", "").strip()
            tests_part = text_data.split("DIAGNOSTIC TESTS:")[1].strip()
            
            # Handle additional sections like TIPS
            if "TIPS:" in tests_part:
                tests_part = tests_part.split("TIPS:")[0].strip()

            # Parse diseases
            for line in diseases_part.split('\n'):
                line = line.strip()
                if line and line.startswith("-"):
                    # Remove the dash and split by " - "
                    content = line.lstrip("- ").strip()
                    if " - " in content:
                        parts = content.split(" - ", 1)
                        name = parts[0].strip()
                        description = parts[1].strip()
                        if name and description:
                            result["diseases"].append({
                                "name": name,
                                "description": description
                            })

            # Parse diagnostic tests
            for line in tests_part.split('\n'):
                line = line.strip()
                if line and line.startswith("-"):
                    # Remove the dash and split by " - "
                    content = line.lstrip("- ").strip()
                    parts = content.split(" - ")
                    
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        description = parts[1].strip()
                        tips = parts[2].strip() if len(parts) > 2 else ""
                        
                        if name and description:
                            result["tests"].append({
                                "name": name,
                                "description": description,
                                "tips": tips
                            })
        
        # If no valid parsing occurred, add default message
        if not result["diseases"] and not result["tests"]:
            result["tests"].append({
                "name": "Parsing Issue",
                "description": "Agent output did not match expected format. Please try again.",
                "tips": ""
            })
                    
    except Exception as e:
        logger.error(f"Error parsing agent output: {e}")
        result["tests"].append({
            "name": "Parsing Error",
            "description": "Error processing diagnostic recommendations.",
            "tips": ""
        })

    return result

# Health check endpoint for AWS load balancer
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        return jsonify({
            "status": "healthy", 
            "service": "medical-diagnostic-api",
            "database": "connected"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "service": "medical-diagnostic-api", 
            "error": "Database connection failed"
        }), 503

# Main API endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    """Main endpoint for medical diagnostic recommendations"""
    try:
        # Get and validate JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400

        # Extract and validate required fields
        symptoms = data.get("symptoms", "").strip()
        vitals = data.get("vitals", "").strip()
        age = data.get("age", "").strip()
        
        if not symptoms or not vitals or not age:
            return jsonify({
                "success": False,
                "error": "Missing required fields: age, symptoms, or vitals are required."
            }), 400

        # Log the request
        logger.info(f"Processing diagnostic request - Age: {age}, Symptoms: {symptoms[:100]}...")

        # Create the medical diagnostic prompt
        prompt = f"""
You are a medical assistant AI that provides diagnostic recommendations based on patient information.

Your task is to analyze the patient's age, symptoms, and vitals to provide:
1. Possible diseases with clear one-line descriptions
2. Diagnostic test recommendations with justifications and safety considerations

STRICT RULES:
- If the input does NOT contain valid medical symptoms, vitals, or age, respond with:
  "I cannot provide diagnostic recommendations without valid medical symptoms, vitals, and age."
- Always follow this EXACT output format:

POSSIBLE DISEASES:
- Disease Name - One line description explaining why this disease matches the symptoms.
- Disease Name - One line description explaining why this disease matches the symptoms.

DIAGNOSTIC TESTS:
- Test Name - Reason for recommendation - Safety considerations or contraindications.
- Test Name - Reason for recommendation - Safety considerations or contraindications.

PATIENT INFORMATION:
Age: {age}
Symptoms: {symptoms}
Vitals: {vitals}

Provide your medical analysis now:
"""

        # Run the agent with the prompt
        logger.info("Running diagnostic agent...")
        agent_output = agent.run(prompt)
        logger.info("Agent completed successfully")
        
        # Parse the agent output
        parsed_output = parse_agent_output(agent_output)
        
        # Log successful completion
        logger.info(f"Diagnostic complete - Found {len(parsed_output['diseases'])} diseases and {len(parsed_output['tests'])} tests")
        
        return jsonify({
            "success": True,
            "diseases": parsed_output["diseases"],
            "tests": parsed_output["tests"],
            "message": "Diagnostic recommendations generated successfully"
        })

    except Exception as e:
        logger.error(f"Diagnostic error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "An internal error occurred while processing your request. Please try again later."
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False, 
        "error": "Endpoint not found. Use POST /recommend for diagnostic recommendations."
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "success": False, 
        "error": "Internal server error"
    }), 500

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed. Use POST for /recommend endpoint."
    }), 405

if __name__ == "__main__":
    try:
        logger.info(f"Starting Medical Diagnostic API on port {port}")
        logger.info("MongoDB Atlas vector store ready")
        logger.info("Available endpoints: POST /recommend, GET /health")
        
        # For production, use a proper WSGI server like Gunicorn
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise