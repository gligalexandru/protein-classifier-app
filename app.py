import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Protein Classifier App",
    page_icon="🧬",
    layout="wide"
)

# App Title and Description
st.title("🧬 Protein Classification App")
st.markdown("""
Welcome to the Protein Classification App! 
Upload a CSV file containing protein data (like `proteinas_test.csv`), and our trained machine learning model will predict the protein classes.
""")

# Load the trained model and label encoder
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_protein_classifier.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, le
    except FileNotFoundError:
        return None, None

model, le = load_models()

if model is None or le is None:
    st.warning("⚠️ Model files not found. Please make sure you have run the Jupyter Notebook to generate `best_protein_classifier.pkl` and `label_encoder.pkl` in this directory.")
    st.stop()

# Define the k-mer extraction function (must match the one used during training)
def get_kmers(sequence, k=3):
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

# File Uploader
st.markdown("### 1. Upload your Data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    
    st.markdown("### 2. Data Preview")
    st.dataframe(df.head())
    
    # Define the required numerical columns expected by the model
    numerical_cols = ['Massa_Molecular', 'Ponto_Isoelétrico', 'Hidrofobicidade', 
                      'Carga_Total', 'Proporção_Polar', 'Proporção_Apolar', 'Comprimento_Sequência']
    
    # Check if all required columns are present in the uploaded file
    missing_cols = [col for col in numerical_cols + ['Sequência'] if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
    else:
        with st.spinner("Processing data and running predictions..."):
            # Prepare data for prediction
            process_df = df.copy()
            
            # Extract k-mers from the sequence
            process_df['kmers'] = process_df['Sequência'].apply(lambda x: get_kmers(str(x), k=3))
            
            # Run predictions
            predictions = model.predict(process_df)
            
            # Decode the numerical predictions back to original string labels
            decoded_predictions = le.inverse_transform(predictions)
            
            # Add predictions to the dataframe
            df['Predicted_Class'] = decoded_predictions
            
            # Try to get prediction probabilities (confidence) if the model supports it
            try:
                probabilities = model.predict_proba(process_df)
                max_probs = probabilities.max(axis=1)
                df['Confidence_Score'] = max_probs
            except AttributeError:
                pass # Model might not support predict_proba (like some SVM configurations)
                
        st.success("✅ Predictions generated successfully!")
        
        # Display the results table
        st.markdown("### 3. Prediction Results")
        st.dataframe(df)
        
        # Interactive Visualizations using Plotly
        st.markdown("### 4. 📊 Interactive Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar Chart: Distribution of Predicted Classes
            class_counts = df['Predicted_Class'].value_counts().reset_index()
            class_counts.columns = ['Predicted Class', 'Count']
            
            fig_bar = px.bar(class_counts, x='Predicted Class', y='Count', 
                             color='Predicted Class', 
                             title="Distribution of Predicted Classes")
            st.plotly_chart(fig_bar, width='stretch')
            
        with col2:
            # Pie Chart: Proportion of Predicted Classes
            fig_pie = px.pie(class_counts, names='Predicted Class', values='Count', 
                             title="Proportion of Predicted Classes", hole=0.4)
            st.plotly_chart(fig_pie, width='stretch')
            
        # Scatter Plot: Feature relationships colored by predicted class
        st.markdown("#### Feature Relationships")
        fig_scatter = px.scatter(df, x='Massa_Molecular', y='Ponto_Isoelétrico', 
                                 color='Predicted_Class', 
                                 hover_data=['ID_Proteína'] if 'ID_Proteína' in df.columns else None,
                                 title="Molecular Mass vs Isoelectric Point (Colored by Predicted Class)",
                                 labels={'Massa_Molecular': 'Molecular Mass', 'Ponto_Isoelétrico': 'Isoelectric Point'})
        st.plotly_chart(fig_scatter, width='stretch')
        
        # Download Button for the results
        st.markdown("### 5. Download Results")
        csv_output = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_output,
            file_name="protein_predictions_results.csv",
            mime="text/csv",
        )
