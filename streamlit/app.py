import streamlit as st
import joblib
import re
import string
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive { color: #17BF63; font-weight: bold; }
    .sentiment-negative { color: #E0245E; font-weight: bold; }
    .sentiment-neutral { color: #FFAD1F; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load('../models/model_final.pkl')
    tfidf = joblib.load('../models/tfidf_vectorizer.pkl')
    le = joblib.load('../models/label_encoder.pkl')
    return model, tfidf, le


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text


def create_gauge_chart(value, title="Confidence"):
    if value >= 0.7:
        color = "#17BF63"
    elif value >= 0.4:
        color = "#FFAD1F"
    else:
        color = "#E0245E"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "steps": [
                {"range": [0, 40], "color": "#fce4ec"},
                {"range": [40, 70], "color": "#fff8e1"},
                {"range": [70, 100], "color": "#e8f5e9"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": value * 100
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_bar_chart(probs, classes):
    colors = ['#E0245E', '#FFAD1F', '#17BF63']
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition='auto',
            textfont=dict(size=14, color='white')
        )
    ])
    
    fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(248,249,250,1)'
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    return fig


def create_radar_chart(probs, classes):
    probs_closed = np.append(probs, probs[0])
    classes_closed = np.append(classes, classes[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=probs_closed,
        theta=classes_closed,
        fill='toself',
        fillcolor='rgba(29, 161, 242, 0.3)',
        line=dict(color='#1DA1F2', width=2),
        marker=dict(size=8, color='#1DA1F2'),
        name='Sentiment'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%'),
            angularaxis=dict(tickfont=dict(size=12))
        ),
        showlegend=False,
        title="Sentiment Radar",
        height=350,
        margin=dict(l=40, r=40, t=50, b=20)
    )
    return fig


def create_treemap(word_data):
    if not word_data:
        return None
    
    labels = [item['name'] for item in word_data]
    values = [item['value'] for item in word_data]
    colors = px.colors.qualitative.Set3[:len(labels)]
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=values,
        marker=dict(colors=colors, line=dict(width=2, color='white')),
        textinfo="label+value",
        textfont=dict(size=14),
        hovertemplate="<b>%{label}</b><br>Frequency: %{value}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Top Words",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def get_wordcloud_data(text):
    words = text.split()
    word_freq = pd.Series(words).value_counts().head(12)
    return [{"name": word, "value": int(freq)} for word, freq in word_freq.items()]


def main():
    st.markdown('<h1 class="main-header">🐦 Twitter Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        model, tfidf, le = load_model()
    except FileNotFoundError:
        st.error("Model files not found!")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.info(
            "Predict sentiment of airline tweets.\n\n"
            "**Model:** LinearSVC (80.36%)\n"
            "**Features:** TF-IDF bigrams\n"
            "**Classes:** Negative, Neutral, Positive"
        )
        st.markdown("## 📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "80.36%")
        with col2:
            st.metric("F1-Score", "0.74")
    
    # Initialize session state
    if 'tweet' not in st.session_state:
        st.session_state.tweet = ""
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Input Tweet")
        
        # Example buttons
        st.markdown("**Try an example:**")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            if st.button("😞 Negative", key="btn_neg"):
                st.session_state.tweet = "@united worst flight ever! Lost my luggage and no one cares."
                st.rerun()
        
        with c2:
            if st.button("😐 Neutral", key="btn_neu"):
                st.session_state.tweet = "@delta just landed safely. Pretty standard flight."
                st.rerun()
        
        with c3:
            if st.button("😊 Positive", key="btn_pos"):
                st.session_state.tweet = "@southwestair amazing service! Best airline ever!"
                st.rerun()
        
        # Text area
        tweet_input = st.text_area(
            "Enter a tweet:",
            value=st.session_state.tweet,
            placeholder="Example: @united worst flight ever!",
            height=150
        )
        st.session_state.tweet = tweet_input
        
        if st.button("🔍 Analyze", type="primary"):
            if tweet_input.strip():
                clean_text = preprocess_text(tweet_input)
                vec = tfidf.transform([clean_text])
                pred = model.predict(vec)
                label = le.inverse_transform(pred)[0]
                
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(vec)[0]
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / exp_scores.sum()
                else:
                    probs = model.predict_proba(vec)[0]
                
                st.session_state.prediction = {
                    'label': label,
                    'probabilities': probs,
                    'clean_text': clean_text
                }
    
    with col2:
        st.markdown("### 📊 Results")
        
        if st.session_state.prediction:
            result = st.session_state.prediction
            icon_map = {'negative': '😞', 'neutral': '😐', 'positive': '😊'}
            icon = icon_map.get(result['label'], '❓')
            
            st.markdown(f"## {icon} {result['label'].upper()}")
            
            max_prob = max(result['probabilities'])
            st.plotly_chart(create_gauge_chart(max_prob), width="stretch")
            
            st.markdown("**Confidence:**")
            classes = ['Negative', 'Neutral', 'Positive']
            for cls, prob in zip(classes, result['probabilities']):
                st.progress(prob, text=f"{cls}: {prob:.1%}")
    
    # Charts
    if st.session_state.prediction:
        st.markdown("---")
        st.markdown("### 📈 Visualizations")
        
        result = st.session_state.prediction
        classes = ['Negative', 'Neutral', 'Positive']
        probs = result['probabilities']
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_bar_chart(probs, classes), width="stretch")
        with c2:
            st.plotly_chart(create_radar_chart(probs, classes), width="stretch")
        
        st.markdown("---")
        st.markdown("### ☁️ Word Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            tweet_text = result['clean_text']
            sentiment = result['label']
            color_map = {'negative': 'Reds', 'neutral': 'YlOrBr', 'positive': 'Greens'}
            
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap=color_map.get(sentiment, 'viridis'),
                max_words=50
            ).generate(tweet_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Word Cloud ({sentiment.upper()})', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        with c2:
            word_data = get_wordcloud_data(tweet_text)
            treemap = create_treemap(word_data)
            if treemap:
                st.plotly_chart(treemap, width="stretch")


if __name__ == "__main__":
    main()
