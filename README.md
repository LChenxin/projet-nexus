# Project Nexus: Enterprise Project Discovery Agent

A Deep-Research-inspired AI agent project for discovering internal initiatives, surfacing external examples, and generating structured recommendations for new enterprise projects.

## Overview

Project Nexus is an AI agent prototype designed to support early-stage project research in enterprise environments.

This project is a small working MVP / proof-of-concept of the core agent workflow. It is intentionally scoped as a demoable prototype rather than a production-ready system.

## Architecture

The system follows a staged agent workflow:

```mermaid
graph TD
    User([User Query]) -->|Role: standard/admin| Init[State Initialization]
    
    Init --> Planner[Planner Agent]
    Planner -->|Early Reject| Rejected([Rejected Output])
    Planner -->|Generates Subtasks| Executor{Executor}
    
    Executor -->|allow_c2 flag| Internal[(Internal Search)]
    Executor -->|External Context| Web[Web Search]
    
    Internal -->|Silent Filtering applied| Evidence((Evidence Pack))
    Web --> Evidence
    
    Evidence --> Summarizer[Summarizer Agent]
    Summarizer -->|Markdown Draft| Guard[Guardrails]
    
    Guard -->|Format/Citations OK| Final([Final Report])
    Guard -->|C2 Leak / Missing Sections| Blocked([Blocked Output])
    
    %% Observability Trace
    Planner -.-> Logger[(Event Logger)]
    Executor -.-> Logger
    Summarizer -.-> Logger
    Guard -.-> Logger
    Logger -.-> Trace[Trace JSON\nfrontend/app.py rendering]
    
    classDef agent fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef tool fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef data fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    
    class Planner,Summarizer agent;
    class Internal,Web tool;
    class Evidence,Trace data;
```

1. **Planner**  
   Converts a user query into structured research subtasks.

2. **Internal Search Tool**  
   Searches a synthetic internal project database using embedding-based semantic retrieval.

3. **Web Search Tool**  
   Retrieves external examples from public web search.

4. **Summarizer**  
   Synthesizes internal and external evidence into a structured report.

5. **Guardrails**  
   Performs lightweight checks on structure, citations, and restricted-content leakage.

6. **Observability Layer**  
   Logs events, warnings, metrics, and trace data for transparency.

## Repository structure

```
Project Nexus/
├── backend/                
│   ├── agent/              
│   │   ├── state.py        
│   │   ├── pipeline.py
│   │   └── prompts.py     
│   ├── tools/              
│   │   ├── internal_search.py 
│   │   └── web_search.py      
│   ├── eval/               
│   │   └── evaluate.py     
│   ├── config.py
│   └── llm.py           
├── frontend/               
│   └── app.py              
├── data/                   
│   └── internal_projects.json 
├── logs/                   
├── .gitignore              
├── README.md               
└── requirements.txt        
```

## Tech stack
- **Backend**: Python,  OpenAI API
- **Frontend**: Streamlit
- **Data**: Synthetic internal project dataset, web search results
- **Testing**: pytest

## How to Run
1. Clone the repository
2. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=900

EMBEDDING_MODEL=text-embedding-3-small
RETRIEVAL_TOP_K=3
RETRIEVAL_THRESHOLD=0.3

ENABLE_WEB_SEARCH=true
WEB_TOP_K=3

INTERNAL_ALLOW_C2=false
```
3. Install dependencies:

```
pip install -r requirements.txt
```
4. Run the Streamlit app:

```
streamlit run frontend/app.py
```
5. Run tests:

```
python -m pytest backend/eval/evaluate.py -v
```
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
