from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Required for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure matplotlib for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


class DrugGraphGenerator:
    """Generate all clinical graphs for drug interactions"""
    
    @staticmethod
    def create_safety_score_chart(interactions_data):
        """Generate bar chart of safety scores"""
        if not interactions_data:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        drug_pairs = [f"{i['drugA']} + {i['drugB']}" for i in interactions_data]
        scores = [94 if i['safety'] == 'safe' else 65 if i['safety'] == 'caution' else 28 for i in interactions_data]
        colors = ['#10b981' if i['safety'] == 'safe' else '#f59e0b' if i['safety'] == 'caution' else '#ef4444' for i in interactions_data]
        
        bars = ax.bar(drug_pairs, scores, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Safety Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Drug Combinations', fontsize=12, fontweight='bold')
        ax.set_title('Clinical Safety Assessment Dashboard', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{score}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add threshold lines
        ax.axhline(y=70, color='#10b981', linestyle='--', alpha=0.5, label='Safe Threshold (70)')
        ax.axhline(y=40, color='#f59e0b', linestyle='--', alpha=0.5, label='Caution Threshold (40)')
        
        # Add legend
        safe_patch = mpatches.Patch(color='#10b981', label='Safe/Low Risk')
        caution_patch = mpatches.Patch(color='#f59e0b', label='Moderate Risk')
        unsafe_patch = mpatches.Patch(color='#ef4444', label='High Risk/Critical')
        ax.legend(handles=[safe_patch, caution_patch, unsafe_patch], loc='upper right')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    @staticmethod
    def create_pie_chart(interactions_data):
        """Generate pie chart for interaction types"""
        if not interactions_data:
            return None
            
        safe_count = sum(1 for i in interactions_data if i['safety'] == 'safe')
        caution_count = sum(1 for i in interactions_data if i['safety'] == 'caution')
        unsafe_count = sum(1 for i in interactions_data if i['safety'] == 'unsafe')
        
        labels = ['Safe/Low Risk', 'Moderate Risk', 'High Risk/Critical']
        sizes = [safe_count, caution_count, unsafe_count]
        colors = ['#10b981', '#f59e0b', '#ef4444']
        explode = (0.05, 0.05, 0.1)
        
        if sum(sizes) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', shadow=True, startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax.set_title('Interaction Risk Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    @staticmethod
    def create_radar_chart(drug_attributes):
        """Generate radar chart for drug comparison"""
        if not drug_attributes or len(drug_attributes) < 2:
            return None
        
        categories = ['Safety', 'Efficacy', 'Metabolism', 'Half-life', 'Interactions']
        num_vars = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899', '#06B6D4']
        
        for idx, drug in enumerate(drug_attributes[:6]):  # Max 6 drugs
            values = drug.get('scores', [80, 85, 75, 70, 80])
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=drug['name'][:15], 
                   color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
        ax.set_title('Drug Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        ax.grid(True)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    @staticmethod
    def create_heatmap(drug_matrix, drug_names):
        """Generate correlation heatmap for drugs"""
        if not drug_matrix or len(drug_names) < 2:
            return None
        
        n = len(drug_names)
        matrix_data = []
        
        for d1 in drug_names:
            row = []
            for d2 in drug_names:
                if d1 == d2:
                    row.append(100)
                else:
                    pair_key = '|'.join(sorted([d1, d2]))
                    row.append(drug_matrix.get(pair_key, 65))
            matrix_data.append(row)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Show all ticks
        ax.set_xticks(np.arange(len(drug_names)))
        ax.set_yticks(np.arange(len(drug_names)))
        ax.set_xticklabels([d[:12] for d in drug_names], rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([d[:12] for d in drug_names], fontsize=9)
        
        # Add text annotations
        for i in range(len(drug_names)):
            for j in range(len(drug_names)):
                text = ax.text(j, i, matrix_data[i][j], ha="center", va="center",
                              color="black" if matrix_data[i][j] > 50 else "white",
                              fontsize=8, fontweight='bold')
        
        ax.set_title('Drug Interaction Heatmap\n(Higher Score = Safer)', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label='Safety Score (0-100)')
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()


@app.route('/')
def index():
    """Serve the main HTML file"""
    return render_template('index.html')


@app.route('/api/generate_graphs', methods=['POST'])
def generate_graphs():
    """API endpoint to generate graphs from interaction data"""
    try:
        data = request.json
        interactions = data.get('interactions', [])
        drug_attributes = data.get('drug_attributes', [])
        history = data.get('history', [])
        drug_matrix = data.get('drug_matrix', {})
        
        graphs = {}
        
        # Generate safety score bar chart
        if interactions:
            safety_chart = DrugGraphGenerator.create_safety_score_chart(interactions)
            if safety_chart:
                graphs['safety_chart'] = safety_chart
            
            pie_chart = DrugGraphGenerator.create_pie_chart(interactions)
            if pie_chart:
                graphs['pie_chart'] = pie_chart
        
        # Generate radar chart if drug attributes available
        if drug_attributes and len(drug_attributes) >= 2:
            radar_chart = DrugGraphGenerator.create_radar_chart(drug_attributes)
            if radar_chart:
                graphs['radar_chart'] = radar_chart
        
        # Generate heatmap
        if drug_matrix:
            drug_names = list(set())
            for key in drug_matrix.keys():
                drugs = key.split('|')
                drug_names.extend(drugs)
            drug_names = sorted(list(set(drug_names)))
            
            if len(drug_names) >= 2:
                heatmap = DrugGraphGenerator.create_heatmap(drug_matrix, drug_names)
                if heatmap:
                    graphs['heatmap'] = heatmap
        
        return jsonify({
            'success': True,
            'graphs': graphs,
            'message': f'Generated {len(graphs)} graphs successfully'
        })
        
    except Exception as e:
        print(f"Error generating graphs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error generating graphs'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Drug Interaction API is running'
    })


if __name__ == '__main__':
    print("=" * 60)
    print(" MediSafe Pro Ultimate - Drug Interaction API")
    print("=" * 60)
    print(" Graph Generator Ready")
    print("🌐 Server running at: http://localhost:5000")
    print(" Endpoint: POST /api/generate_graphs")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)