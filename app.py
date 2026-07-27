import os
import streamlit as st
from dotenv import load_dotenv

# Dependencias de LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_cohere import CohereEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Configuración visual orientada a industrias latinoamericanas (colores de seguridad)
st.set_page_config(
    page_title="Agente de Respuesta Rápida: Accidentes Industriales",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados para dar un aspecto de tablero de control industrial
st.markdown("""
<style>
    .stApp {
        background-color: #f4f6f9; /* Fondo gris claro industrial */
    }
    .main-header {
        color: #d32f2f; /* Rojo alerta/peligro */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
        text-transform: uppercase;
        border-bottom: 3px solid #ff9800; /* Naranja precaución */
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 5px solid #1976d2; /* Azul seguridad */
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        border-left: 5px solid #d32f2f; /* Rojo para el asistente (alerta) */
    }
    .sidebar .sidebar-content {
        background-color: #37474f; /* Gris oscuro para panel de control */
        color: white;
    }
    div[data-testid="stSidebarNav"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Cargar variables de entorno (solo necesario en local, en Streamlit se usa st.secrets)
load_dotenv()

# Función para obtener API keys de forma segura (soporta local .env y Streamlit Secrets)
def get_api_key(key_name):
    api_key = os.environ.get(key_name)
    if api_key:
        return api_key
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return None

COHERE_API_KEY = get_api_key("COHERE_API_KEY")
GROQ_API_KEY = get_api_key("GROQ_API_KEY")

if not COHERE_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ Faltan las claves API. Por favor, configura COHERE_API_KEY y GROQ_API_KEY.")
    st.stop()

DOCS_DIR = "documentos-pdfs"
CHROMA_PATH = "chroma_db"
# Usamos un modelo rápido y potente para emergencias
LLM_MODEL = "llama-3.3-70b-versatile"

st.markdown('<h1 class="main-header">⚠️ Sistema de Respuesta Rápida: Incidentes con Materiales y Sustancias Peligrosas</h1>', unsafe_allow_html=True)
st.markdown("**Agente entrenado con manuales de actuación técnica para emergencias químicas e industriales.**")

# Inicializar Embeddings (Cohere es excelente para texto y multilingüismo)
@st.cache_resource
def get_embeddings():
    return CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=COHERE_API_KEY
    )

embeddings = get_embeddings()

# Inicializar LLM (Groq para latencia ultra baja, vital en emergencias)
@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0.1, # Baja temperatura para respuestas factuales
        model_name=LLM_MODEL,
        groq_api_key=GROQ_API_KEY
    )

llm = get_llm()

# Función para cargar documentos, fragmentarlos y crear/cargar la base vectorial
@st.cache_resource(show_spinner="Preparando base de conocimiento (esto puede tomar un momento la primera vez)...")
def setup_vectorstore():
    # 1. Carga de Documentos desde Directorio
    # Se indica la carga de TODOS los PDFs en la carpeta
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        st.warning(f"Se creó la carpeta '{DOCS_DIR}'. Por favor, añade los manuales PDF.")
        st.stop()
        
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()
    
    if not documents:
        st.error(f"No se encontraron documentos en '{DOCS_DIR}'.")
        st.stop()

    # Si la base de datos ya existe, simplemente cárgala (para no recalcular embeddings)
    if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
        return vectorstore

    # 2. Fragmentación Semántica (Chunking)
    # Se utiliza SemanticChunker que divide el texto basándose en la similitud de significado
    text_splitter = SemanticChunker(embeddings)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Almacenamiento Vectorial Local (Chroma)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return vectorstore

vectorstore = setup_vectorstore()

# 4. Creación del Recuperador (Retriever)
# Se buscan los 4 fragmentos más relevantes (k=4)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# === IMPLEMENTACIÓN DE REESCRITURA DE CONSULTA (Query Rewriting) ===
# Útil en emergencias donde los usuarios escriben rápido, mal o de forma ambigua
rewrite_template = """
Eres un despachador de emergencias industriales experto. 
Tu tarea es tomar la consulta inicial de un operario bajo estrés y reescribirla de manera formal, 
clara y técnica para que nuestro sistema de búsqueda de documentos técnicos pueda encontrar los 
procedimientos exactos.

Consulta original del operario: {question}

Reescribe la consulta enfocándote en identificar:
1. El tipo de sustancia involucrada (si se menciona).
2. El tipo de incidente (derrame, fuego, exposición, etc.).
3. La acción requerida (primeros auxilios, contención, evacuación).

Devuelve ÚNICAMENTE la consulta reescrita. No agregues saludos ni explicaciones.
Consulta Reescrita:"""

rewrite_prompt = PromptTemplate.from_template(rewrite_template)

# Cadena para reescribir la pregunta
query_rewriter = rewrite_prompt | llm | StrOutputParser()


# === PROMPT TEMPLATE PRINCIPAL ===
# Plantilla estricta para asegurar que responda solo basándose en el contexto
qa_template = """
Eres un Agente de Actuación Técnica en Emergencias Industriales y Materiales Peligrosos (HAZMAT).
Tu objetivo principal es proporcionar protocolos de acción inmediatos, precisos y seguros.

Utiliza ÚNICAMENTE los siguientes fragmentos de los manuales técnicos recuperados para responder a la pregunta.
Si la información necesaria NO está en los fragmentos proporcionados, indica claramente: 
"ALERTA: La información específica solicitada no se encuentra en los manuales de actuación actuales. Proceda con precaución estándar y contacte a especialistas."
NO inventes información ni asumas procedimientos de seguridad que no estén en el texto.

Fragmentos técnicos recuperados:
{context}

Pregunta del usuario (situación):
{question}

Formato de Respuesta Requerido:
1. **Prioridad Inmediata**: (La acción más urgente a tomar)
2. **Procedimientos Específicos**: (Lista de pasos a seguir según los manuales)
3. **Peligros Identificados**: (Riesgos a tener en cuenta)

Respuesta Técnica:
"""

prompt = PromptTemplate.from_template(qa_template)

# Función para formatear los documentos recuperados en un solo texto
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# CREACIÓN DE LA CADENA PRINCIPAL CON LANGCHAIN (LCEL)
# cadena = contexto_y_pregunta | prompt | llm | parser
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Interfaz de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if user_query := st.chat_input("Describa la emergencia (ej. 'Derrame de ácido sulfúrico en zona B...'):"):
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando situación y recuperando protocolos..."):
            try:
                # 1. Reescritura de la consulta
                optimized_query = query_rewriter.invoke({"question": user_query})
                st.caption(f"*(Consulta optimizada internamente: {optimized_query})*")
                
                # 2. Ejecución de la Cadena Principal (INVOKE)
                # Aquí llamamos a la cadena con la consulta optimizada
                response = rag_chain.invoke(optimized_query)
                
                st.markdown(response)
                # Guardar en historial
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error crítico al procesar la solicitud: {str(e)}")

# Información lateral
with st.sidebar:
    st.markdown("### 📚 Manuales Cargados")
    st.info("""
    - Sustancias químicas corrosivas y tóxicas
    - Compuestos metálicos peligrosos y metales pesados
    - Hidrocarburos peligrosos y derivados del petróleo
    - Plaguicidas y fitosanitarios
    - Gases tóxicos, corrosivos y asfixiantes
    - Reactivos explosivos, organoperóxidos y sustancias inestables
    """)
    st.divider()
    st.markdown("**Motor RAG:** LangChain")
    st.markdown("**LLM:** Groq (Llama-3.3-70b-versatile)")
    st.markdown("**Embeddings:** Cohere")
    st.markdown("**Vector DB:** Chroma (Local)")