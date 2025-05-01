# Pet Health Data Analysis Project

## Project Overview
This project analyzes health data of domestic animals (cats and dogs) to identify risk factors associated with various diseases. The study focuses on understanding how physiological and demographic characteristics influence pets' overall health conditions, with applications in veterinary health informatics and predictive analytics.


# Project Structure
```markdown
DATA_ANALYSIS/
|-- analysis/                 # Virtual environment
|   |-- Include/
|   |-- Lib/
|   |-- Scripts/
|   |-- share/
|   |-- pyvenv.cfg
|-- data/                     # Data directory
|   |-- cleaned_data.xlsx     # Processed dataset
|-- deployment/               # Deployment files
|   |-- gbm_model.pkl         # Trained Gradient Boosting Model
|   |-- scaler.pkl            # Feature scaler object
|-- app.py                    # Main application file
|-- requirements.txt          # Project dependencies
|-- README.md                 # Project documentation
```
## Dataset Description
The dataset contains comprehensive information on domestic animals including:
- Species and breed
- Age and weight
- Vital signs (temperature, respiratory rate, heart rate)
- Activity level
- Sleep quality
- Disease diagnosis

## Research Problem
**How do the physiological and demographic characteristics of domestic animals influence their health status, and can we identify at-risk profiles for certain diseases?**

## Project Objectives
1. Identify factors most correlated with various diseases
2. Determine breed/species predispositions to health conditions
3. Analyze impact of age and weight on pet health
4. Visualize complex feature relationships
5. Build typical profiles for healthy and sick animals
6. Generate personalized health recommendations

## Installation & Usage

### Prerequisites
- Python 3.x
- pip package manager

### Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pet-health-analysis.git
   cd pet-health-analysis
2. **Set up virtual environment**:
    ```bash
    python -m venv analysis
    # On Windows:
    analysis\Scripts\activate
    # On macOS/Linux:
    source analysis/bin/activate
4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
6. **Run the application**:
    ```bash
    python app.py
## Dependencies
The project utilizes the following Python libraries:

- **pandas** - Data manipulation and analysis  
- **numpy** - Numerical computations  
- **matplotlib** - Static data visualizations  
- **seaborn** - Advanced statistical visualizations  
- **scikit-learn** - Machine learning and clustering  
- **missingno** - Missing data visualization  
- **plotly** - Interactive visualizations  
- **prince** - Multiple correspondence analysis (for categorical variable reduction)  
- **streamlit** - Interactive web dashboard (if included)  

## Authors

**Mabrouka Messaoudi**  
📧 Email: [mabrouka.messaoudi@fss.u-sfax.tn](mailto:mabrouka.messaoudi@fss.u-sfax.tn)  

**Essra Hmida**  
📧 Email: [hmidaesraa@gmail.com](mailto:hmidaesraa@gmail.com)  




