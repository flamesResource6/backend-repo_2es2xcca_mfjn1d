from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Portfolio API", version="1.0.0")

# Allow CORS for local dev and any frontend preview URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "OK", "service": "Portfolio API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test")
def test():
    # Simple diagnostics without requiring a real database
    return {
        "backend": "FastAPI running",
        "database": "not used",
        "database_url": None,
        "database_name": None,
        "connection_status": "n/a",
        "collections": []
    }

@app.get("/profile")
def get_profile():
    return {
        "name": "Adarsh Kesharwani",
        "contacts": {
            "email": "akesharwani900@gmail.com",
            "phone": "+91 7577897882",
            "portfolio": "https://adarshhme.vercel.app/",
            "github": "https://github.com/Adarshh9",
            "linkedin": "https://www.linkedin.com/in/adarsh-kesharwani-4b2146261",
            "medium": "https://medium.com/@adarshhme",
            "kaggle": "https://kaggle.com/adarsh926"
        },
        "experience": [
            {
                "company": "Anyway AI",
                "role": "Gen AI Intern",
                "location": "Remote",
                "duration": "Dec 2024 – Aug 2025",
                "highlights": [
                    "Achieved 15% mAP improvement in YOLOv8 by curating high-quality synthetic data using Stable Diffusion fine-tuned via LoRA, DreamBooth, and ControlNet.",
                    "Broadened model applicability across 8+ industrial domains by generating and benchmarking task-specific synthetic datasets to significantly enhance robustness.",
                    "Reduced data labeling time by 40% through automation scripts integrated into the synthetic data generation workflow.",
                    "Deployed an internal chatbot leveraging Langchain and FastAPI, streamlining intra-team data documentation and experiment tracking."
                ]
            },
            {
                "company": "Vasana Technologies",
                "role": "AI Intern",
                "location": "Remote",
                "duration": "Jun 2024 – Jul 2024",
                "highlights": [
                    "Built a 3D object generation pipeline from 2D images using TripoSR and NeRF, reducing manual modeling time by 40%.",
                    "Deployed via Docker and AWS Lambda, achieving 30% cost savings through serverless scaling and faster inference turnaround."
                ]
            }
        ],
        "projects": [
            {
                "name": "FinSaathi – LLM Finance Advisor",
                "url": "https://github.com/Adarshh9/Finsaathi_Datathon",
                "date": "Feb 2025",
                "highlights": [
                    "Delivered 40% faster financial insight generation by engineering an LLM-powered (DeepSeek-R1-Distill-Llama-70B) advisory chatbot with Elasticsearch-driven retrieval.",
                    "Implemented Monte Carlo simulations for portfolio risk assessment, connected via yfinance API for real-time data ingestion."
                ]
            },
            {
                "name": "TaxoCapsNet – Hierarchical Bio-Classifier",
                "url": "https://github.com/Adarshh9/TaxoCapsNet",
                "date": "Aug 2025",
                "highlights": [
                    "Achieved 96% accuracy and 0.98 ROC-AUC in microbiome classification by developing a hierarchical Capsule Network aligned with biological taxonomy.",
                    "Enabled interpretable diagnostics through SHAP analysis, linking model predictions to biologically relevant microbial taxa."
                ]
            },
            {
                "name": "NutriAI – Graph-based Diet Recommender",
                "url": "https://github.com/Adarshh9/NutriAI",
                "date": "May 2025",
                "highlights": [
                    "Built a personalized dietary recommender leveraging GNNs over a 10K+ relationship knowledge graph, achieving 90% relevance retention through RL-based caching.",
                    "Deployed FastAPI + Redis inference pipeline for real-time personalized nutrition insights under 20ms latency."
                ]
            },
            {
                "name": "ProdML – Production MLOps Stack",
                "url": "https://github.com/Adarshh9/ProdML-InIt",
                "date": "Oct 2025",
                "highlights": [
                    "Delivered production-grade image classification platform handling 279 req/s throughput and <10ms p50 latency via Redis caching and optimized I/O.",
                    "Integrated MLflow for experiment tracking, CI/CD for automated deployment, and observability via Prometheus and Grafana.",
                    "Extended platform to mobile (React Native + PyTorch Mobile), enabling real-time on-device inference with full offline support."
                ]
            }
        ],
        "education": {
            "institution": "Thakur College of Engineering and Technology",
            "degree": "B.Tech in AI and Data Science",
            "location": "Mumbai, India",
            "duration": "Nov 2022 – May 2026 (expected)",
            "gpa": "8.3"
        },
        "skills": {
            "programming_data": ["Python", "SQL", "JavaScript", "Pandas", "NumPy", "Flask", "REST APIs", "Data Pipelines", "Streaming (Kafka basics)", "Efficient Vectorized Computation"],
            "llm_agents": ["LLM Fine-tuning (LoRA, QLoRA, PEFT)", "RAG Systems (LangChain, LlamaIndex)", "RLHF", "Prompt Engineering", "Multi-Agent Systems", "Function Calling", "VectorDB Integration (FAISS, Chroma)"],
            "generative_visual": ["Stable Diffusion", "ControlNet", "DreamBooth", "NeRF", "Text-to-Image/3D Synthesis", "Neural Audio Synthesis", "Diffusion Model Optimization", "Data Augmentation Pipelines"],
            "dl_ml_frameworks": ["PyTorch", "TensorFlow", "Transformers", "YOLOv8", "OpenCV", "Scikit-learn", "GNNs", "ONNX", "AutoML (NAS)", "PyTorch Mobile", "Expo"],
            "mlops_deployment": ["AWS", "GCP", "Docker", "FastAPI", "Redis", "MLflow", "DVC", "W&B", "GitHub Actions (CI/CD)", "Prometheus", "Grafana", "API Observability", "Model Quantization"]
        },
        "achievements": [
            "Winner – AMD AI Sprint 2025",
            "Winner – Datazen Datathon 2025"
        ],
        "certifications": [
            "Machine Learning Specialization",
            "Data, ML and AI in Google Cloud",
            "Engineer Data in Google Cloud",
            "Postman Student Expert"
        ]
    }
