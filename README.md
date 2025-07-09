# Spirits Market Analysis and Forecasting System

An advanced Agentic AI system for analyzing market data, generating forecasts, collecting market insights, simulating outcomes, and creating comprehensive, actionable reports.

## Project Overview

This project implements a sophisticated analysis pipeline that using multiple AI agents and LLM Models to provide deep insights into spirits market trends. It streamlines the complex process of data preparation, forecasting, external news integration, scenario simulation, and automated reporting, enabling business consultants to derive meaningful conclusions efficiently.

## Architecture and Key Concepts

The system employs a hybrid architecture, combining a custom Python-based Orchestrator with a LangGraph-inspired design for agent interactions, ensuring a modular, robust, and extensible framework.

### 1. Orchestrator (`src/main.py`)
- Acts as the central controller, managing the overall flow of the analysis pipeline.
- Initializes and coordinates the execution of various specialized agents.
- Handles configuration loading, logging, and error management across the pipeline.
- Passes data and results sequentially between agents, ensuring a clear flow of information.

### 2. Specialized Agents (`src/agents/`, `src/data_processing/data_handler.py`)
Each agent is a distinct Python class with a focused task, leveraging LLMs for intelligent decision-making, analysis, and summarization:

- **Data Preparation Agent (`src/data_processing/data_handler.py`):**
    - **LLM-Driven Data Understanding:** Analyzes raw data (columns, types, nulls, statistics) and uses an LLM to suggest optimal null value treatment and feature engineering strategies.
    - **Comprehensive Null Value Treatment:** Implements various imputation techniques (mean, median, mode, ffill, bfill, drop) based on LLM recommendations.
    - **Advanced Feature Engineering:** Creates new features (e.g., time-based, lag, rolling statistics) to enrich the dataset for forecasting, guided by LLM suggestions.
    - **Anomaly Detection and Treatment:** Identifies and treats outliers using methods like IQR or Isolation Forest to ensure a clean, high-quality dataset.
- **Forecasting Agent (`src/agents/forecasting_agent.py`):**
    - **LLM-Selected Models:** Utilizes an LLM to recommend the most suitable forecasting model (Prophet, SARIMAX, RandomForest) based on data characteristics and business context.
    - **Sales Forecasting:** Generates future sales predictions with confidence intervals.
    - **Unusual Trend Detection:** Proactively identifies significant incremental or decremental trends in forecasts, highlighting potential market shifts.
- **News Agent (`src/agents/news_agent.py`):**
    - **API-First News Collection:** Fetches relevant market news using external APIs (e.g., NewsAPI - placeholder integration, requires API key) with a fallback to web scraping if the API is unavailable or not configured.
    - **LLM-Powered Summarization:** Summarizes collected news and explains its potential relation to forecasted trends.
- **Simulation Agent (`src/agents/simulation_agent.py`):**
    - **LLM-Generated What-If Scenarios:** Uses an LLM to propose various business scenarios (e.g., price changes, promotions, competitive actions).
    - **Dynamic Simulation Logic:** Simulates outcomes based on calculated price elasticity from historical data, providing more realistic impact assessments.
    - **LLM-Summarized Insights:** Summarizes simulation results, identifying beneficial/risky scenarios and recommending actionable strategies.
- **Summary Agent (`src/agents/summary_agent.py`):**
    - **LLM-Generated Executive Summary:** Combines insights from forecasting (including trend analysis), news, and simulation into a concise, actionable executive summary and key headlines.
    - **Automated Presentation Generation:** Creates professional PowerPoint presentations (`.pptx`) with charts and key findings from all agents, including detailed forecast trends and simulation outcomes.

### 3. LLM Integration (`utils/llm_utils.py`)
- A flexible utility module for interacting with various LLM providers (e.g., OpenAI, Gemini - with placeholders for future integration).
- Centralizes LLM configuration (provider, model, API key) from `config.yaml`.

### 4. Communication and State Management
- Agents communicate primarily through explicit function calls, passing processed data and intermediate results.
- The system is designed to integrate with LangGraph for stateful multi-agent workflows, where a shared `state` dictionary (`Dict[str, Any]`) is passed and updated between agent nodes.

### 5. Tool Use
- Wrappers are implemented for external interactions:
    - **LLM APIs:** Handled by `utils/llm_utils.py`.
    - **News APIs/Web Scraping:** Managed by `NewsAgent`.
    - **File I/O:** Handled by `DataHandler` (CSV) and `utils/ppt_utils.py` (PPTX, images).

## Project Structure

```
.
├── config/
│   └── config.yaml           # Main configuration file (updated with LLM and News API keys)
├── data/
│   ├── us_spirits_purchase_data.csv # Sample spirits purchase data
│   ├── products.csv          # Product master data
│   └── clusters.csv          # Customer cluster data
├── src/
│   ├── data_processing/
│   │   └── data_handler.py   # Centralized data loading, preprocessing, feature engineering, anomaly detection (LLM-guided)
│   ├── agents/
│   │   ├── forecasting_agent.py # Handles sales forecasting and trend detection (LLM-powered model selection)
│   │   ├── news_agent.py        # Collects and summarizes market news (API-first, LLM-powered summary)
│   │   ├── simulation_agent.py  # Runs what-if scenarios and impact assessments (LLM-generated scenarios, elasticity-based simulation)
│   │   └── summary_agent.py     # Generates executive summaries and reports (LLM-powered summary, PPT generation)
│   └── main.py                 # Orchestrates the entire analysis pipeline
├── utils/
│   ├── llm_utils.py          # LLM API interaction utility (flexible provider support)
│   └── ppt_utils.py          # PowerPoint report generation utility (enhanced for trends and simulations)
├── output/
│   ├── processed_data/       # (Not explicitly saved by current DataHandler, but conceptually here)
│   ├── forecasts/
│   ├── news/
│   ├── simulations/
│   ├── reports/
│   └── presentations/        # Generated PowerPoint reports will be saved here
├── logs/                     # Application logs
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Features

1.  **Sophisticated Data Preparation**
    *   LLM-driven data understanding for intelligent null treatment suggestions.
    *   Automated and configurable null value imputation using various techniques (mean, median, mode, ffill, bfill).
    *   Dynamic feature engineering (time-based, lag features, rolling statistics) to enhance predictive power.
    *   Robust anomaly detection and treatment (IQR, Isolation Forest) for cleaner data.

2.  **Intelligent Forecasting**
    *   LLM-guided selection of optimal forecasting models (Prophet, SARIMAX, RandomForest).
    *   Generation of future sales forecasts with confidence intervals.
    *   Automated detection and reporting of unusual incremental or decremental trends in forecasts.

3.  **Contextual Market Insights**
    *   Integration with external News APIs (with a simple web scraping fallback) for real-time market news collection.
    *   LLM-powered summarization of news articles, relating them to forecasted trends and providing market context.

4.  **Dynamic Scenario Simulation**
    *   LLM-generated "what-if" scenarios for various market interventions (e.g., price changes, promotions).
    *   Simulation of outcomes using calculated price elasticity based on historical data.
    *   Analysis of cross-category impacts and consumer segments (conceptual, extensible).

5.  **Automated & Rich Reporting**
    *   LLM-generated executive summaries and key headlines, combining all insights.
    *   Automatic generation of professional PowerPoint presentations (`.pptx`) with:
        *   Forecast charts and detailed trend analysis.
        *   News highlights and sentiment.
        *   Comprehensive simulation results, including simulated sales and impact assessments.
    *   Centralized logging for pipeline execution monitoring.

## Setup Instructions

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd NIQ_Hackfest # Or your project directory name
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The dependencies in `requirements.txt` are selected for compatibility with recent Python versions (including Python 3.12.10). If you encounter any specific version conflicts during installation, consider upgrading pip or checking for a more specific version of the problematic library.*

4.  **Configure the project**
    -   Open `config/config.yaml`.
    -   **Add your API Keys:**
        -   Under the `news` section, set `api_key: YOUR_NEWS_API_KEY` for a news API (e.g., NewsAPI). If not provided, the system will fall back to basic Google News scraping.
        -   Under the new `llm` section, set `api_key: YOUR_OPENAI_API_KEY` (or the API key for your chosen LLM provider). You can also configure `provider` (e.g., `openai`, `gemini`) and `model` (e.g., `gpt-4`, `gemini-pro`).
    -   Review and update other configuration parameters (data paths, forecasting horizons, simulation scenarios) as needed.

5.  **Prepare the data**
    -   Place your primary data files (`xyz.csv`) in the `data/` directory.
    -   Ensure the file names match those specified in `config/config.yaml`.

## Running the Analysis

To run the complete agentic AI analysis pipeline, execute the `main.py` script:

```bash
python src/main.py
```

The system will:
1.  Load and preprocess data using the LLM-guided `DataHandler`.
2.  Generate forecasts and detect unusual trends using the `ForecastingAgent`.
3.  Collect and summarize market news using the `NewsAgent`.
4.  Run "what-if" simulations using the `SimulationAgent`.
5.  Generate a comprehensive executive summary and a PowerPoint report using the `SummaryAgent`.

Progress and critical information will be logged to the console and to log files in the `logs/` directory.

## Output

The system generates various outputs in the `output/` directory:

-   `output/presentations/`: The generated PowerPoint report (`.pptx` file) containing the executive summary, key headlines, forecast charts, trend analysis, news summary, and detailed simulation insights.
-   Log files in the `logs/` directory.

## Configuration

The `config/config.yaml` file is central to customizing the system. It contains settings for:
-   Data paths and processing parameters (including LLM-guided data prep).
-   Forecasting model selection and parameters.
-   News collection search terms and API keys.
-   Simulation scenarios and their parameters.
-   Report generation options and output paths.
-   **LLM Configuration**: Allows selection of LLM provider, model, and API key.
-   Logging verbosity and format.

## Contributing

1.  Fork the repository.
2.  Create a feature branch for your contributions.
3.  Commit your changes following good commit message practices.
4.  Push your changes to your feature branch.
5.  Create a Pull Request to the main repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue in the repository.
