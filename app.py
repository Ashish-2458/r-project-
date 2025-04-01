from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import datetime
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Set matplotlib backend to avoid Tk dependency
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import calendar
from collections import defaultdict

# Initialize Flask app
app = Flask(__name__)

# Constants for prediction
DEFAULT_CYCLE_LENGTH = 28
LUTEAL_PHASE_AVG = 14

# Define Python models to replace R functions
class MenstrualCycleModel:
    """Model to predict cycle length"""
    def __init__(self):
        self.model = LinearRegression()
        # Pre-fit the model with average values
        X = np.array([[28, 14, 14], [30, 15, 14], [26, 13, 13], [32, 16, 15]])
        y = np.array([28, 30, 26, 32])
        self.model.fit(X, y)
    
    def predict(self, length_of_cycle, estimated_day_of_ovulation, length_of_luteal_phase):
        # Create features array
        X = np.array([[length_of_cycle, estimated_day_of_ovulation, length_of_luteal_phase]])
        # Predict cycle length
        return int(round(self.model.predict(X)[0]))

class SymptomPredictionModel:
    """Model to predict menstrual symptoms based on user parameters"""
    def __init__(self):
        # Load the dataset and create simple predictive model
        try:
            df = pd.read_csv('period - Copy.csv')
            # Create a simple scoring system
            df['symptom_score'] = df['Menses_score'] * 1.5
            
            # Train a random forest model
            features = ['Age', 'Length_of_cycle', 'BMI', 'Unusual_Bleeding']
            df['Unusual_Bleeding'] = df['Unusual_Bleeding'].replace({'yes': 1, 'no': 0})
            
            # Get the features and target
            X = df[features].copy()
            y = df['symptom_score']
            
            # Train the model
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.model.fit(X, y)
            
            # Feature importance
            self.feature_importance = dict(zip(features, self.model.feature_importances_))
            
        except Exception as e:
            print(f"Error initializing symptom model: {e}")
            # Fallback to a simple model
            self.model = None
            
    def predict_symptoms(self, age, cycle_length, bmi, unusual_bleeding):
        """Predict symptom severity and characteristics"""
        if self.model is None:
            # Fallback logic if model couldn't be trained
            base_score = 5
            # Adjust based on heuristics
            if unusual_bleeding:
                base_score += 2
            if bmi > 25 or bmi < 18.5:
                base_score += 1
            if cycle_length > 35 or cycle_length < 25:
                base_score += 1
                
            return min(base_score, 10)
        else:
            # Use the trained model
            features = np.array([[age, cycle_length, bmi, unusual_bleeding]])
            prediction = self.model.predict(features)[0]
            return min(round(prediction, 1), 10)
    
    def get_common_symptoms(self, severity):
        """Return common symptoms based on severity score"""
        symptoms = {
            "mild": ["Mild cramping", "Light fatigue", "Minor mood changes"],
            "moderate": ["Moderate cramping", "Fatigue", "Mood swings", "Bloating", "Headaches"],
            "severe": ["Severe cramping", "Significant fatigue", "Mood changes", "Bloating", 
                      "Headaches/Migraines", "Nausea", "Dizziness"]
        }
        
        if severity < 4:
            return symptoms["mild"]
        elif severity < 7:
            return symptoms["moderate"]
        else:
            return symptoms["severe"]

class FertilityPredictor:
    """Predicts fertile window and ovulation days"""
    
    def predict_fertility_window(self, last_period_date, cycle_length, luteal_phase):
        """
        Calculate fertility window
        
        Args:
            last_period_date: First day of last period (datetime.date)
            cycle_length: Average cycle length in days (int)
            luteal_phase: Length of luteal phase in days (int)
            
        Returns:
            Dictionary with fertility window dates and ovulation date
        """
        # Convert string to date if needed
        if isinstance(last_period_date, str):
            last_period_date = datetime.datetime.strptime(last_period_date, '%Y-%m-%d').date()
        
        # Calculate ovulation day (cycle length - luteal phase)
        ovulation_day = cycle_length - luteal_phase
        
        # Calculate dates
        ovulation_date = last_period_date + datetime.timedelta(days=ovulation_day)
        
        # Fertility window (5 days before ovulation + ovulation day)
        fertile_start = ovulation_date - datetime.timedelta(days=5)
        fertile_end = ovulation_date
        
        # High fertility days (3 days before ovulation + ovulation day)
        high_fertile_start = ovulation_date - datetime.timedelta(days=3)
        
        # Format dates
        result = {
            'ovulation_date': ovulation_date.strftime('%Y-%m-%d'),
            'fertile_window_start': fertile_start.strftime('%Y-%m-%d'),
            'fertile_window_end': fertile_end.strftime('%Y-%m-%d'),
            'high_fertility_start': high_fertile_start.strftime('%Y-%m-%d'),
        }
        
        return result

def predict_menstrual_health(
    length_of_cycle,
    estimated_day_of_ovulation,
    length_of_luteal_phase,
    length_of_menses,
    unusual_bleeding,
    bmi,
    menses_score,
    age
):
    """Function to assess menstrual health based on parameters"""
    
    # Calculate PCOD risk (enhanced with more parameters)
    pcod_risk = 10  # baseline risk
    
    # BMI and cycle length are key indicators
    if bmi > 25 and length_of_cycle > 35:
        pcod_risk = 75
    elif bmi > 25 or length_of_cycle > 35:
        pcod_risk = 50
    elif unusual_bleeding == 1:
        pcod_risk = 30
        
    # Adjust based on luteal phase (atypical luteal phase can indicate hormonal issues)
    if length_of_luteal_phase < 10 or length_of_luteal_phase > 16:
        pcod_risk += 10
        
    # Very long or short cycles at younger ages more concerning
    if age < 25 and (length_of_cycle > 38 or length_of_cycle < 22):
        pcod_risk += 10
        
    # Cap at 100%
    pcod_risk = min(pcod_risk, 100)
    
    # Determine cycle irregularity
    irregular = "Yes" if (length_of_cycle < 25 or length_of_cycle > 35 or
                         length_of_luteal_phase < 10 or length_of_luteal_phase > 16) else "No"
    
    # Calculate health score (1-10)
    health_score = 10 - (menses_score/2 + 
                        (2 if unusual_bleeding == 1 else 0) +
                        (1.5 if irregular == "Yes" else 0))
    health_score = max(1, min(10, round(health_score, 1)))
    
    # Generate personalized recommendations
    recommendations = generate_health_recommendations(
        pcod_risk, 
        irregular, 
        health_score, 
        length_of_cycle, 
        unusual_bleeding,
        bmi
    )
    
    return {
        "PCOD_Risk_Score": f"{pcod_risk}%",
        "Cycle_Irregularity": irregular,
        "Overall_Health_Score": f"{health_score}/10",
        "Recommendations": recommendations
    }

def generate_health_recommendations(pcod_risk, irregular, health_score, cycle_length, unusual_bleeding, bmi):
    """Generate personalized health recommendations"""
    recommendations = []
    
    # PCOD risk recommendations
    if pcod_risk >= 60:
        recommendations.append("Consider consulting with a gynecologist to discuss PCOD/PCOS evaluation.")
    elif pcod_risk >= 30:
        recommendations.append("Monitor your menstrual cycle regularly and discuss any concerns with a healthcare provider.")
    
    # Cycle irregularity recommendations
    if irregular == "Yes":
        recommendations.append("Track your cycle with a period app to better understand your pattern.")
        if cycle_length > 35:
            recommendations.append("Long cycles may benefit from regular exercise and a balanced diet.")
        elif cycle_length < 25:
            recommendations.append("Short cycles may indicate hormonal fluctuations; consider stress reduction techniques.")
    
    # Unusual bleeding recommendations
    if unusual_bleeding:
        recommendations.append("Unusual bleeding should be discussed with a healthcare provider.")
    
    # BMI recommendations
    if bmi > 25:
        recommendations.append("Maintaining a healthy weight through balanced nutrition and regular activity can help regulate cycles.")
    elif bmi < 18.5:
        recommendations.append("Being underweight can affect menstruation; consider consulting with a nutritionist.")
    
    # General recommendations
    if health_score < 5:
        recommendations.append("Consider anti-inflammatory foods and supplements like omega-3 fatty acids to help with symptoms.")
    
    recommendations.append("Stay hydrated and consider iron-rich foods during menstruation.")
    
    return recommendations

def generate_calendar_data(last_period_date, cycle_length, period_length):
    """Generate 3-month calendar with period and fertility predictions"""
    
    # Convert to date object if string
    if isinstance(last_period_date, str):
        last_period_date = datetime.datetime.strptime(last_period_date, '%Y-%m-%d').date()
    
    # Create fertility predictor
    fertility_predictor = FertilityPredictor()
    
    # Calculate next 3 periods
    calendar_data = {}
    current_date = last_period_date
    
    for i in range(3):
        # Period dates
        period_start = current_date
        period_end = period_start + datetime.timedelta(days=period_length-1)
        
        # Fertility window
        fertility_data = fertility_predictor.predict_fertility_window(
            period_start, 
            cycle_length, 
            14  # Assuming average luteal phase
        )
        
        # Next period start
        next_period = period_start + datetime.timedelta(days=cycle_length)
        
        # Add to calendar data
        month_key = period_start.strftime('%Y-%m')
        if month_key not in calendar_data:
            calendar_data[month_key] = {
                'periods': [],
                'fertile_windows': [],
                'ovulation_days': []
            }
        
        # Add data
        calendar_data[month_key]['periods'].append({
            'start': period_start.strftime('%Y-%m-%d'),
            'end': period_end.strftime('%Y-%m-%d')
        })
        
        calendar_data[month_key]['fertile_windows'].append({
            'start': fertility_data['fertile_window_start'],
            'end': fertility_data['fertile_window_end']
        })
        
        calendar_data[month_key]['ovulation_days'].append(
            fertility_data['ovulation_date']
        )
        
        # Move to next cycle
        current_date = next_period
    
    return calendar_data

def format_calendar_for_display(calendar_data):
    """Format calendar data for template display"""
    formatted_data = []
    
    for month_key, data in calendar_data.items():
        year, month = month_key.split('-')
        month_name = calendar.month_name[int(month)]
        
        # Get all dates in month
        _, num_days = calendar.monthrange(int(year), int(month))
        
        # Create day data
        days = []
        for day in range(1, num_days+1):
            date_str = f"{year}-{month}-{day:02d}"
            
            # Check if this day is in period
            in_period = any(
                period['start'] <= date_str <= period['end'] 
                for period in data['periods']
            )
            
            # Check if this day is in fertile window
            in_fertile = any(
                window['start'] <= date_str <= window['end'] 
                for window in data['fertile_windows']
            )
            
            # Check if this is ovulation day
            is_ovulation = date_str in data['ovulation_days']
            
            # Determine day class
            day_class = ''
            if in_period:
                day_class = 'period-day'
            elif in_fertile:
                day_class = 'fertile-day'
                if is_ovulation:
                    day_class = 'ovulation-day'
            
            days.append({
                'day': day,
                'class': day_class
            })
        
        # Add to formatted data
        formatted_data.append({
            'month_name': f"{month_name} {year}",
            'days': days
        })
    
    return formatted_data

# Initialize models
cycle_model = MenstrualCycleModel()
symptom_model = SymptomPredictionModel()
fertility_predictor = FertilityPredictor()

# Save models to disk for persistence (optional)
def save_models():
    with open('menstrual_cycle_model.pkl', 'wb') as f:
        pickle.dump(cycle_model, f)
    print("Models saved to disk")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_next_cycle', methods=['POST'])
def predict_next_cycle():
    try:
        # Get form data
        last_period_date = request.form['last_period_date']
        length_of_cycle = int(request.form['length_of_cycle'])
        estimated_day_of_ovulation = int(request.form['estimated_day_of_ovulation'])
        length_of_luteal_phase = int(request.form['length_of_luteal_phase'])
        length_of_menses = int(request.form.get('length_of_menses', 5))
        
        # Predict using the Python model
        predicted_cycle_length = cycle_model.predict(
            length_of_cycle, 
            estimated_day_of_ovulation, 
            length_of_luteal_phase
        )
        
        # Calculate next period date
        last_period_date_obj = datetime.datetime.strptime(last_period_date, '%Y-%m-%d').date()
        next_period_date = last_period_date_obj + datetime.timedelta(days=predicted_cycle_length)
        
        # Get fertility window
        fertility_data = fertility_predictor.predict_fertility_window(
            last_period_date_obj,
            length_of_cycle,
            length_of_luteal_phase
        )
        
        # Generate calendar data
        calendar_data = generate_calendar_data(
            last_period_date_obj,
            length_of_cycle,
            length_of_menses
        )
        
        # Format calendar for display
        formatted_calendar = format_calendar_for_display(calendar_data)
        
        return render_template('result.html', 
                              prediction_type="Next Cycle",
                              next_period_date=next_period_date.strftime('%Y-%m-%d'),
                              fertility_data=fertility_data,
                              calendar_data=formatted_calendar)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/predict_health', methods=['POST'])
def predict_health():
    try:
        # Get form data
        length_of_cycle = int(request.form['length_of_cycle'])
        estimated_day_of_ovulation = int(request.form['estimated_day_of_ovulation'])
        length_of_luteal_phase = int(request.form['length_of_luteal_phase'])
        length_of_menses = int(request.form['length_of_menses'])
        unusual_bleeding = int(request.form['unusual_bleeding'])
        bmi = float(request.form['bmi'])
        menses_score = int(request.form['menses_score'])
        age = int(request.form.get('age', 25))  # Default to 25 if not provided
        
        # Call the prediction function
        results = predict_menstrual_health(
            length_of_cycle,
            estimated_day_of_ovulation,
            length_of_luteal_phase,
            length_of_menses,
            unusual_bleeding,
            bmi,
            menses_score,
            age
        )
        
        # Predict symptoms
        symptom_severity = symptom_model.predict_symptoms(
            age, 
            length_of_cycle, 
            bmi, 
            unusual_bleeding
        )
        
        # Get common symptoms
        common_symptoms = symptom_model.get_common_symptoms(symptom_severity)
        
        # Add to results
        results['symptom_severity'] = f"{symptom_severity}/10"
        results['common_symptoms'] = common_symptoms
        
        return render_template('result.html',
                              prediction_type="Health Assessment",
                              pcod_risk=results["PCOD_Risk_Score"],
                              cycle_irregularity=results["Cycle_Irregularity"],
                              health_score=results["Overall_Health_Score"],
                              recommendations=results["Recommendations"],
                              symptom_severity=results["symptom_severity"],
                              common_symptoms=results["common_symptoms"])
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/analysis')
def analysis():
    try:
        # Load the dataset
        df = pd.read_csv('period - Copy.csv')
        
        # Convert categorical
        df['Unusual_Bleeding'] = df['Unusual_Bleeding'].map({'yes': 1, 'no': 0})
        
        # Basic statistics
        stats = {
            'total_records': len(df),
            'avg_cycle_length': round(df['Length_of_cycle'].mean(), 2),
            'avg_ovulation_day': round(df['Estimated_day_of_ovulution'].mean(), 2),
            'avg_luteal_phase': round(df['Length_of_Leutal_Phase'].mean(), 2),
            'avg_menses_length': round(df['Length_of_menses'].mean(), 2),
            'unusual_bleeding_count': df['Unusual_Bleeding'].sum(),
            'unusual_bleeding_percent': round(df['Unusual_Bleeding'].mean() * 100, 2),
            'min_cycle': df['Length_of_cycle'].min(),
            'max_cycle': df['Length_of_cycle'].max(),
        }
        
        # Advanced statistics
        # Calculate regularity - standard deviation of cycle length
        regularity = df.groupby('Age')['Length_of_cycle'].std().mean()
        stats['cycle_regularity'] = round(regularity, 2)
        
        # Create visualizations
        plots = []
        
        # 1. Cycle Length Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df['Length_of_cycle'], kde=True)
        plt.title('Distribution of Menstrual Cycle Lengths')
        plt.xlabel('Cycle Length (days)')
        plt.ylabel('Frequency')
        plots.append(get_plot_url())
        
        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        numeric_cols = df.select_dtypes(include=['number']).columns
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Between Menstrual Parameters')
        plt.tight_layout()
        plots.append(get_plot_url())
        
        # 3. BMI vs Cycle Length
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='BMI', y='Length_of_cycle', hue='Unusual_Bleeding')
        plt.title('BMI vs Cycle Length')
        plt.xlabel('BMI')
        plt.ylabel('Cycle Length (days)')
        plots.append(get_plot_url())
        
        # 4. Age vs Cycle Length
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Age', y='Length_of_cycle')
        plt.title('Age vs Cycle Length')
        plt.xlabel('Age')
        plt.ylabel('Cycle Length (days)')
        plt.xticks(rotation=90)
        plots.append(get_plot_url())
        
        # 5. Menses Score Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Menses_score', data=df)
        plt.title('Distribution of Menses Scores')
        plt.xlabel('Menses Score')
        plt.ylabel('Count')
        plots.append(get_plot_url())
        
        # 6. Luteal Phase Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df['Length_of_Leutal_Phase'], kde=True)
        plt.title('Distribution of Luteal Phase Lengths')
        plt.xlabel('Luteal Phase Length (days)')
        plt.ylabel('Frequency')
        plots.append(get_plot_url())
        
        # 7. Relationship between Luteal Phase and Cycle Length
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='Length_of_cycle', y='Length_of_Leutal_Phase')
        plt.title('Relationship between Cycle Length and Luteal Phase')
        plt.xlabel('Cycle Length (days)')
        plt.ylabel('Luteal Phase Length (days)')
        plots.append(get_plot_url())
        
        # 8. Ovulation day patterns
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Unusual_Bleeding', y='Estimated_day_of_ovulution')
        plt.title('Ovulation Day by Unusual Bleeding')
        plt.xlabel('Unusual Bleeding')
        plt.ylabel('Estimated Day of Ovulation')
        plots.append(get_plot_url())
        
        # Generate insights
        insights = generate_insights(df, stats)
        
        return render_template('analysis.html', 
                              stats=stats, 
                              plots=plots, 
                              insights=insights)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/tracker')
def tracker():
    """Cycle tracking calendar view"""
    return render_template('tracker.html')

@app.route('/resources')
def resources():
    """Educational resources on menstrual health"""
    return render_template('resources.html')

@app.route('/api/predict_next_cycle', methods=['POST'])
def api_predict_next_cycle():
    try:
        data = request.get_json()
        
        # Extract data from JSON
        last_period_date = data['last_period_date']
        length_of_cycle = int(data['length_of_cycle'])
        estimated_day_of_ovulation = int(data['estimated_day_of_ovulation'])
        length_of_luteal_phase = int(data['length_of_luteal_phase'])
        
        # Predict using the Python model
        predicted_cycle_length = cycle_model.predict(
            length_of_cycle, 
            estimated_day_of_ovulation, 
            length_of_luteal_phase
        )
        
        # Calculate next period date
        last_period_date_obj = datetime.datetime.strptime(last_period_date, '%Y-%m-%d').date()
        next_period_date = last_period_date_obj + datetime.timedelta(days=predicted_cycle_length)
        
        # Get fertility window
        fertility_data = fertility_predictor.predict_fertility_window(
            last_period_date_obj,
            length_of_cycle,
            length_of_luteal_phase
        )
        
        response = {
            'next_period_date': next_period_date.strftime('%Y-%m-%d'),
            'predicted_cycle_length': predicted_cycle_length,
            'fertility_window': {
                'start': fertility_data['fertile_window_start'],
                'end': fertility_data['fertile_window_end']
            },
            'ovulation_date': fertility_data['ovulation_date']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_health', methods=['POST'])
def api_predict_health():
    try:
        data = request.get_json()
        
        # Extract data from JSON
        length_of_cycle = int(data['length_of_cycle'])
        estimated_day_of_ovulation = int(data['estimated_day_of_ovulation'])
        length_of_luteal_phase = int(data['length_of_luteal_phase'])
        length_of_menses = int(data['length_of_menses'])
        unusual_bleeding = int(data['unusual_bleeding'])
        bmi = float(data['bmi'])
        menses_score = int(data['menses_score'])
        age = int(data.get('age', 25))  # Default to 25 if not provided
        
        # Call the prediction function
        results = predict_menstrual_health(
            length_of_cycle,
            estimated_day_of_ovulation,
            length_of_luteal_phase,
            length_of_menses,
            unusual_bleeding,
            bmi,
            menses_score,
            age
        )
        
        # Add symptom prediction
        symptom_severity = symptom_model.predict_symptoms(
            age, 
            length_of_cycle, 
            bmi, 
            unusual_bleeding
        )
        
        results['symptom_severity'] = f"{symptom_severity}/10"
        results['common_symptoms'] = symptom_model.get_common_symptoms(symptom_severity)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_plot_url():
    """Convert matplotlib plot to base64 image for HTML display"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return plot_url

def generate_insights(df, stats):
    """Generate insights based on the dataset"""
    insights = []
    
    # Cycle length insights
    if stats['avg_cycle_length'] > 32:
        insights.append("The average cycle length in the dataset is slightly longer than typical. " +
                        "Longer cycles may be associated with hormonal fluctuations.")
    elif stats['avg_cycle_length'] < 26:
        insights.append("The average cycle length in the dataset is shorter than typical. " +
                        "Short cycles can sometimes be associated with lower estrogen levels.")
    else:
        insights.append("The average cycle length falls within the typical range of 26-32 days, " +
                      "suggesting overall regular menstrual patterns in the population.")
    
    # Luteal phase insights
    if stats['avg_luteal_phase'] < 10:
        insights.append("The average luteal phase length is shorter than optimal. " +
                      "A short luteal phase may affect fertility and can sometimes be addressed with lifestyle changes.")
    elif stats['avg_luteal_phase'] > 16:
        insights.append("The average luteal phase is longer than typical. " +
                      "This may affect cycle predictability.")
    
    # BMI and cycle correlation
    corr = df[['BMI', 'Length_of_cycle']].corr().iloc[0,1]
    if abs(corr) > 0.2:
        if corr > 0:
            insights.append(f"There appears to be a positive correlation ({corr:.2f}) between BMI and cycle length, " +
                          "suggesting BMI may influence cycle duration.")
        else:
            insights.append(f"There appears to be a negative correlation ({corr:.2f}) between BMI and cycle length, " +
                          "suggesting higher BMI may be associated with shorter cycles.")
    
    # Age insights
    age_cycle_corr = df[['Age', 'Length_of_cycle']].corr().iloc[0,1]
    if abs(age_cycle_corr) > 0.15:
        insights.append(f"Age appears to have a {abs(age_cycle_corr):.2f} correlation with cycle length, " +
                      "suggesting cycle patterns may change with age.")
    
    # Unusual bleeding insights
    if stats['unusual_bleeding_percent'] > 15:
        insights.append(f"The dataset shows {stats['unusual_bleeding_percent']}% of cycles involve unusual bleeding, " +
                      "which is higher than expected and may warrant further investigation.")
    
    return insights

if __name__ == '__main__':
    app.run(debug=True) 