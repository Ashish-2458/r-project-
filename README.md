# Menstrual Health Prediction Application

This application provides tools for predicting the next menstrual cycle and assessing menstrual health based on various parameters.

## Features

1. **Next Cycle Prediction**: Predicts when your next period will start based on your menstrual cycle data
2. **Health Assessment**: Evaluates menstrual health metrics including:
   - PCOD (Polycystic Ovary Syndrome) risk assessment
   - Cycle irregularity detection
   - Overall menstrual health score
3. **Data Analysis**: Provides visualizations and statistics from the dataset:
   - Cycle length distribution
   - Correlation between menstrual parameters
   - BMI and age effects on cycle length
   - Menstrual discomfort score distribution

## Setup Instructions

### Prerequisites

- Python 3.8+

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd menstrual-health-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Flask app:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Model Information

The application uses Python-based models:

1. **Cycle Length Prediction Model**: A linear regression model that predicts cycle length based on current cycle parameters
2. **Menstrual Health Assessment Function**: An algorithm that evaluates menstrual health indicators

## API Endpoints

The application exposes the following API endpoints:

1. **POST /api/predict_next_cycle**
   - Predicts the next period start date
   - Required parameters: `last_period_date`, `length_of_cycle`, `estimated_day_of_ovulation`, `length_of_luteal_phase`

2. **POST /api/predict_health**
   - Assesses menstrual health
   - Required parameters: `length_of_cycle`, `estimated_day_of_ovulation`, `length_of_luteal_phase`, `length_of_menses`, `unusual_bleeding`, `bmi`, `menses_score`

## Input Parameter Ranges

For accurate predictions, use values within these ranges:

- **Length of Cycle**: 21-35 days
- **Day of Ovulation**: 12-16 days
- **Luteal Phase**: 12-16 days
- **Length of Menses**: 3-7 days
- **Unusual Bleeding**: 0 (No) or 1 (Yes)
- **BMI**: 18.5-30
- **Menses Score**: 1-10 (1 = minimal discomfort, 10 = severe discomfort)

## Disclaimer

This application is for informational purposes only and is not a substitute for professional medical advice. Always consult with a healthcare provider for medical concerns.

## License

[MIT License](LICENSE) 