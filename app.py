"""
Student Performance Analyzer - Python API
==========================================
A Flask REST API that provides ML-powered student analytics.
Designed to be called from Oracle APEX via APEX_WEB_SERVICE.

Endpoints:
  GET  /api/health          - Health check
  POST /api/predict          - Predict next semester grade using Linear Regression
  POST /api/trend            - Analyze score trends across semesters
  POST /api/chart/performance - Generate a performance radar chart (base64 PNG)
  POST /api/chart/trend       - Generate a trend line chart (base64 PNG)
  POST /api/analyze_all       - All-in-one: prediction + trend + charts

Deploy to: Render.com (free tier)
Author: Student Performance Analyzer Project
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
import traceback
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from APEX


# ============================================================
# ENDPOINT 1: Health Check
# ============================================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check to verify the API is running."""
    return jsonify({
        "status": "healthy",
        "service": "Student Performance Analyzer",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": ["predict", "trend", "chart"]
    })


# ============================================================
# ENDPOINT 2: ML Prediction
# ============================================================
@app.route('/api/predict', methods=['POST'])
def predict_grade():
    """
    Predict the next semester grade using Linear Regression.
    
    This is REAL machine learning - it uses sklearn's LinearRegression
    to find patterns between attendance, past scores, and subject
    difficulty to predict future performance.
    
    Expected JSON input:
    {
        "student_name": "John Smith",
        "scores": [
            {"subject": "Math", "score": 85, "attendance": 92, "semester": "2024-Spring"},
            {"subject": "Physics", "score": 78, "attendance": 88, "semester": "2024-Fall"},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'scores' not in data:
            return jsonify({"error": "Missing 'scores' in request body"}), 400
        
        scores = data['scores']
        student_name = data.get('student_name', 'Unknown')
        
        if len(scores) < 2:
            return jsonify({"error": "Need at least 2 score records for prediction"}), 400
        
        df = pd.DataFrame(scores)
        
        # Feature engineering
        # Convert semester to numeric order
        semester_order = {s: i for i, s in enumerate(
            sorted(df['semester'].unique())
        )}
        df['semester_num'] = df['semester'].map(semester_order)
        
        # Encode subjects as numeric
        subject_map = {s: i for i, s in enumerate(df['subject'].unique())}
        df['subject_num'] = df['subject'].map(subject_map)
        
        # Features: attendance, semester progression, subject
        X = df[['attendance', 'semester_num', 'subject_num']].values
        y = df['score'].values
        
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for next semester (semester_num + 1)
        next_semester = max(df['semester_num']) + 1
        avg_attendance = df['attendance'].mean()
        
        predictions_by_subject = {}
        for subject, subject_num in subject_map.items():
            pred_input = np.array([[avg_attendance, next_semester, subject_num]])
            predicted_score = float(model.predict(pred_input)[0])
            # Clamp between 0 and 100
            predicted_score = max(0, min(100, predicted_score))
            predictions_by_subject[subject] = round(predicted_score, 1)
        
        # Overall prediction
        predicted_avg = np.mean(list(predictions_by_subject.values()))
        
        # Determine grade
        if predicted_avg >= 90:
            grade = 'A'
        elif predicted_avg >= 80:
            grade = 'B'
        elif predicted_avg >= 70:
            grade = 'C'
        elif predicted_avg >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        # Risk assessment
        current_avg = df['score'].mean()
        score_trend = predicted_avg - current_avg
        
        if predicted_avg < 60:
            risk_level = 'Critical'
        elif predicted_avg < 70 and avg_attendance < 80:
            risk_level = 'High'
        elif predicted_avg < 75 or avg_attendance < 85:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Confidence based on R-squared and data quantity
        r_squared = model.score(X, y)
        data_factor = min(1.0, len(scores) / 10)  # More data = higher confidence
        confidence = round(float(r_squared * 0.7 + data_factor * 0.3), 2)
        confidence = max(0.1, min(1.0, confidence))
        
        # Generate recommendation
        recommendations = []
        if risk_level in ('Critical', 'High'):
            recommendations.append("URGENT: Student needs immediate academic intervention.")
        if avg_attendance < 80:
            recommendations.append(f"Attendance is low ({avg_attendance:.0f}%). Improving attendance could boost scores significantly.")
        if score_trend < -5:
            recommendations.append(f"Scores are declining (trend: {score_trend:+.1f} points). Investigate causes.")
        elif score_trend > 5:
            recommendations.append(f"Positive trend detected ({score_trend:+.1f} points). Keep up the good work!")
        
        # Subject-specific advice
        worst_subject = min(predictions_by_subject, key=predictions_by_subject.get)
        best_subject = max(predictions_by_subject, key=predictions_by_subject.get)
        if predictions_by_subject[worst_subject] < 70:
            recommendations.append(f"Focus extra tutoring on {worst_subject} (predicted: {predictions_by_subject[worst_subject]}).")
        recommendations.append(f"Strongest subject: {best_subject} (predicted: {predictions_by_subject[best_subject]}).")
        
        if not recommendations:
            recommendations.append("Student is performing well across all subjects.")
        
        return jsonify({
            "student_name": student_name,
            "predicted_grade": grade,
            "predicted_average": round(predicted_avg, 1),
            "predictions_by_subject": predictions_by_subject,
            "risk_level": risk_level,
            "confidence_score": confidence,
            "score_trend": round(score_trend, 1),
            "current_average": round(current_avg, 1),
            "average_attendance": round(avg_attendance, 1),
            "recommendation": " | ".join(recommendations),
            "model_r_squared": round(r_squared, 3),
            "ml_method": "Linear Regression (sklearn)"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ============================================================
# ENDPOINT 3: Trend Analysis
# ============================================================
@app.route('/api/trend', methods=['POST'])
def analyze_trend():
    """
    Analyze score trends across semesters using statistical methods.
    
    Expected JSON input:
    {
        "scores": [
            {"subject": "Math", "score": 85, "attendance": 92, "semester": "2024-Spring"},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        scores = data.get('scores', [])
        
        if len(scores) < 2:
            return jsonify({"error": "Need at least 2 records for trend analysis"}), 400
        
        df = pd.DataFrame(scores)
        
        # Sort semesters chronologically
        semesters_sorted = sorted(df['semester'].unique())
        semester_map = {s: i for i, s in enumerate(semesters_sorted)}
        df['semester_num'] = df['semester'].map(semester_map)
        
        # Overall trend per semester
        semester_avgs = df.groupby(['semester', 'semester_num']).agg({
            'score': 'mean',
            'attendance': 'mean'
        }).reset_index().sort_values('semester_num')
        
        # Linear regression for trend line
        X_trend = semester_avgs['semester_num'].values.reshape(-1, 1)
        y_trend = semester_avgs['score'].values
        
        trend_model = LinearRegression()
        trend_model.fit(X_trend, y_trend)
        
        slope = float(trend_model.coef_[0])
        
        if slope > 3:
            trend_direction = "Strong Improvement"
        elif slope > 0.5:
            trend_direction = "Slight Improvement"
        elif slope > -0.5:
            trend_direction = "Stable"
        elif slope > -3:
            trend_direction = "Slight Decline"
        else:
            trend_direction = "Significant Decline"
        
        # Per-subject trends
        subject_trends = {}
        for subject in df['subject'].unique():
            subj_df = df[df['subject'] == subject].sort_values('semester_num')
            if len(subj_df) >= 2:
                scores_list = subj_df['score'].tolist()
                change = scores_list[-1] - scores_list[0]
                subject_trends[subject] = {
                    "scores_by_semester": {
                        row['semester']: round(row['score'], 1)
                        for _, row in subj_df.iterrows()
                    },
                    "change": round(change, 1),
                    "direction": "up" if change > 2 else ("down" if change < -2 else "stable")
                }
        
        # Semester-by-semester summary
        semester_summary = []
        for _, row in semester_avgs.iterrows():
            semester_summary.append({
                "semester": row['semester'],
                "avg_score": round(row['score'], 1),
                "avg_attendance": round(row['attendance'], 1)
            })
        
        # Statistical summary
        stats = {
            "overall_mean": round(float(df['score'].mean()), 1),
            "overall_std": round(float(df['score'].std()), 1),
            "highest_score": round(float(df['score'].max()), 1),
            "lowest_score": round(float(df['score'].min()), 1),
            "total_records": len(df),
            "total_semesters": len(semesters_sorted),
            "total_subjects": len(df['subject'].unique())
        }
        
        return jsonify({
            "trend_direction": trend_direction,
            "trend_slope": round(slope, 2),
            "semester_summary": semester_summary,
            "subject_trends": subject_trends,
            "statistics": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ============================================================
# ENDPOINT 4: Performance Radar Chart
# ============================================================
@app.route('/api/chart/performance', methods=['POST'])
def chart_performance():
    """
    Generate a radar/spider chart showing performance across subjects.
    Returns a base64-encoded PNG image.
    
    Expected JSON input:
    {
        "student_name": "John Smith",
        "scores": [
            {"subject": "Math", "score": 85, "attendance": 92},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        scores = data.get('scores', [])
        student_name = data.get('student_name', 'Student')
        
        if not scores:
            return jsonify({"error": "No scores provided"}), 400
        
        df = pd.DataFrame(scores)
        
        # Aggregate by subject (latest or average)
        subject_avgs = df.groupby('subject').agg({
            'score': 'mean',
            'attendance': 'mean'
        }).reset_index()
        
        subjects = subject_avgs['subject'].tolist()
        score_values = subject_avgs['score'].tolist()
        attend_values = subject_avgs['attendance'].tolist()
        
        N = len(subjects)
        if N < 3:
            # For fewer than 3 subjects, use bar chart instead
            return _generate_bar_chart(subjects, score_values, attend_values, student_name)
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        score_values += score_values[:1]
        attend_values += attend_values[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Plot scores
        ax.plot(angles, score_values, 'o-', linewidth=2.5, color='#00d4ff',
                label='Score', markersize=8)
        ax.fill(angles, score_values, alpha=0.15, color='#00d4ff')
        
        # Plot attendance
        ax.plot(angles, attend_values, 's-', linewidth=2.5, color='#ff6b6b',
                label='Attendance', markersize=8)
        ax.fill(angles, attend_values, alpha=0.1, color='#ff6b6b')
        
        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(subjects, size=11, color='white', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'],
                           size=9, color='#888888')
        ax.spines['polar'].set_color('#333366')
        ax.grid(color='#333366', linestyle='-', linewidth=0.5)
        ax.tick_params(colors='white')
        
        # Title and legend
        plt.title(f'{student_name} - Performance Radar',
                  size=16, color='white', fontweight='bold', pad=20)
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                          facecolor='#16213e', edgecolor='#333366',
                          fontsize=11)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#1a1a2e', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            "chart_type": "radar",
            "image_base64": img_base64,
            "mime_type": "image/png"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


def _generate_bar_chart(subjects, scores, attendance, student_name):
    """Fallback bar chart when fewer than 3 subjects."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    x = np.arange(len(subjects))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, scores, width, label='Score',
                   color='#00d4ff', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, attendance, width, label='Attendance',
                   color='#ff6b6b', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Subject', color='white', fontsize=12)
    ax.set_ylabel('Percentage', color='white', fontsize=12)
    ax.set_title(f'{student_name} - Performance', color='white',
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.set_ylim(0, 110)
    ax.spines['bottom'].set_color('#333366')
    ax.spines['left'].set_color('#333366')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#333366', linestyle='--', alpha=0.5)
    
    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', color='#00d4ff', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', color='#ff6b6b', fontsize=10)
    
    legend = ax.legend(facecolor='#16213e', edgecolor='#333366', fontsize=11)
    for text in legend.get_texts():
        text.set_color('white')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({
        "chart_type": "bar",
        "image_base64": img_base64,
        "mime_type": "image/png"
    })


# ============================================================
# ENDPOINT 5: Trend Line Chart
# ============================================================
@app.route('/api/chart/trend', methods=['POST'])
def chart_trend():
    """
    Generate a trend line chart showing score progression over semesters.
    Returns a base64-encoded PNG image.
    
    Expected JSON input:
    {
        "student_name": "John Smith",
        "scores": [
            {"subject": "Math", "score": 85, "attendance": 92, "semester": "2024-Spring"},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        scores = data.get('scores', [])
        student_name = data.get('student_name', 'Student')
        
        if len(scores) < 2:
            return jsonify({"error": "Need at least 2 records for trend chart"}), 400
        
        df = pd.DataFrame(scores)
        semesters_sorted = sorted(df['semester'].unique())
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        fig.patch.set_facecolor('#1a1a2e')
        
        colors = ['#00d4ff', '#ff6b6b', '#ffd93d', '#6bcb77', '#c084fc', '#fb923c']
        
        # Top chart: Score trends by subject
        ax1.set_facecolor('#1a1a2e')
        subjects = df['subject'].unique()
        
        for i, subject in enumerate(subjects):
            subj_df = df[df['subject'] == subject].copy()
            subj_df['sem_order'] = subj_df['semester'].map(
                {s: j for j, s in enumerate(semesters_sorted)}
            )
            subj_df = subj_df.sort_values('sem_order')
            
            color = colors[i % len(colors)]
            ax1.plot(subj_df['semester'], subj_df['score'],
                    'o-', linewidth=2.5, color=color, label=subject,
                    markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            
            # Add score labels
            for _, row in subj_df.iterrows():
                ax1.annotate(f'{row["score"]:.0f}',
                           (row['semester'], row['score']),
                           textcoords="offset points", xytext=(0, 12),
                           ha='center', color=color, fontsize=10, fontweight='bold')
        
        # Overall average trend line
        sem_avg = df.groupby('semester')['score'].mean()
        sem_avg = sem_avg.reindex(semesters_sorted)
        ax1.plot(semesters_sorted, sem_avg.values, '--', linewidth=3,
                color='white', alpha=0.6, label='Overall Avg')
        
        # Grade threshold lines
        thresholds = [(90, 'A', '#27ae60'), (80, 'B', '#2ecc71'),
                      (70, 'C', '#f39c12'), (60, 'D', '#e74c3c')]
        for threshold, label, color in thresholds:
            ax1.axhline(y=threshold, linestyle=':', alpha=0.3, color=color)
            ax1.text(len(semesters_sorted)-0.8, threshold+1, f'{label} ({threshold})',
                    color=color, alpha=0.6, fontsize=9)
        
        ax1.set_ylabel('Score', color='white', fontsize=13, fontweight='bold')
        ax1.set_title(f'{student_name} - Score Trend Analysis',
                     color='white', fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylim(0, 105)
        ax1.tick_params(colors='white', labelsize=10)
        ax1.spines['bottom'].set_color('#333366')
        ax1.spines['left'].set_color('#333366')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.grid(axis='y', color='#333366', linestyle='--', alpha=0.3)
        
        legend = ax1.legend(loc='upper left', facecolor='#16213e',
                           edgecolor='#333366', fontsize=10, ncol=3)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Bottom chart: Attendance trend
        ax2.set_facecolor('#1a1a2e')
        attend_avg = df.groupby('semester')['attendance'].mean()
        attend_avg = attend_avg.reindex(semesters_sorted)
        
        bar_colors = ['#ff6b6b' if v < 80 else '#ffd93d' if v < 90 else '#6bcb77'
                      for v in attend_avg.values]
        
        bars = ax2.bar(semesters_sorted, attend_avg.values, color=bar_colors,
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        
        for bar, val in zip(bars, attend_avg.values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', color='white', fontsize=11,
                    fontweight='bold')
        
        ax2.axhline(y=80, linestyle='--', color='#ff6b6b', alpha=0.5)
        ax2.text(len(semesters_sorted)-0.8, 81, 'Min Target 80%',
                color='#ff6b6b', alpha=0.7, fontsize=9)
        
        ax2.set_ylabel('Attendance %', color='white', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 110)
        ax2.tick_params(colors='white', labelsize=10)
        ax2.spines['bottom'].set_color('#333366')
        ax2.spines['left'].set_color('#333366')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout(pad=2)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#1a1a2e', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({
            "chart_type": "trend",
            "image_base64": img_base64,
            "mime_type": "image/png"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ============================================================
# ENDPOINT 6: All-in-One Analysis
# ============================================================
@app.route('/api/analyze_all', methods=['POST'])
def analyze_all():
    """
    Comprehensive analysis: prediction + trend + both charts.
    Single API call returns everything APEX needs.
    
    Expected JSON input:
    {
        "student_name": "John Smith",
        "scores": [
            {"subject": "Math", "score": 85, "attendance": 92, "semester": "2024-Spring"},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'scores' not in data:
            return jsonify({"error": "Missing 'scores' in request body"}), 400
        
        scores = data['scores']
        student_name = data.get('student_name', 'Unknown')
        
        result = {
            "student_name": student_name,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": None,
            "trend": None,
            "chart_performance": None,
            "chart_trend": None,
            "errors": []
        }
        
        # Run prediction
        try:
            with app.test_request_context(
                '/api/predict', method='POST',
                json=data, content_type='application/json'
            ):
                pred_response = predict_grade()
                if isinstance(pred_response, tuple):
                    result["errors"].append(f"Prediction: {pred_response[0].get_json()}")
                else:
                    result["prediction"] = pred_response.get_json()
        except Exception as e:
            result["errors"].append(f"Prediction error: {str(e)}")
        
        # Run trend analysis
        try:
            with app.test_request_context(
                '/api/trend', method='POST',
                json=data, content_type='application/json'
            ):
                trend_response = analyze_trend()
                if isinstance(trend_response, tuple):
                    result["errors"].append(f"Trend: {trend_response[0].get_json()}")
                else:
                    result["trend"] = trend_response.get_json()
        except Exception as e:
            result["errors"].append(f"Trend error: {str(e)}")
        
        # Generate performance chart
        try:
            with app.test_request_context(
                '/api/chart/performance', method='POST',
                json=data, content_type='application/json'
            ):
                chart_response = chart_performance()
                if isinstance(chart_response, tuple):
                    result["errors"].append(f"Performance chart: {chart_response[0].get_json()}")
                else:
                    result["chart_performance"] = chart_response.get_json().get("image_base64")
        except Exception as e:
            result["errors"].append(f"Performance chart error: {str(e)}")
        
        # Generate trend chart
        try:
            with app.test_request_context(
                '/api/chart/trend', method='POST',
                json=data, content_type='application/json'
            ):
                trend_chart_response = chart_trend()
                if isinstance(trend_chart_response, tuple):
                    result["errors"].append(f"Trend chart: {trend_chart_response[0].get_json()}")
                else:
                    result["chart_trend"] = trend_chart_response.get_json().get("image_base64")
        except Exception as e:
            result["errors"].append(f"Trend chart error: {str(e)}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ============================================================
# Run the app
# ============================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
