import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import random
import streamlit as st

st.set_page_config(page_title="ABSA Tourism Review", page_icon="🏝️",
                   layout='centered', initial_sidebar_state="expanded")

# Load your ensemble model and labels
ensemble_model = pickle.load(open('deploy/ensemble.pkl', 'rb'))
labels = ['attractions_negative',
          'attractions_neutral',
          'attractions_positive',
          'amenities_negative',
          'amenities_neutral',
          'amenities_positive',
          'accessibility_negative',
          'accessibility_neutral',
          'accessibility_positive',
          'image_negative',
          'image_neutral',
          'image_positive',
          'price_negative',
          'price_neutral',
          'price_positive',
          'human_resources_negative',
          'human_resources_neutral',
          'human_resources_positive']

aspects = {
    "attractions": "🔮",
    "amenities": "🛎️",
    "accessibility": "🏄‍♂️",
    "image": "🖼️",
    "price": "💰",
    "human_resources": "👨‍💼"
}

colors = {
    'positive': '#0FBA5D',
    'neutral': '#FEAC00',
    'negative': '#FB576D'
}

def predict(ensemble_model, sentence, labels):
    '''
    Given a sentence and an ensemble model,
    return the label of the sentence.
    '''
    aspects = np.unique([label.split('_')[0] for label in labels])

    try:
        stacked_output = ensemble_model([sentence], mean_pool=True)
        logits = stacked_output.logits.detach().cpu().numpy()[0]
        dict_ = dict(zip(labels, logits))

        label = []
        for aspect in aspects:
            list_key_value = [(key, value) for key, value in dict_.items()
                              if aspect in key.lower()]
            numpy_array_value = np.array([kv[1] for kv in list_key_value])

            if np.any(numpy_array_value > 0):
                id = numpy_array_value.argmax()
                label.append(list_key_value[id][0])

        if len(label) == 0:
            id = logits.argmax()
            label.append(labels[id])

        return label
    except Exception as e:
        print(e)

#! input text
def extract_aspect_sentiment(labels):
    aspect = next((label.split('_')[0] if label.split('_')[0] != "human" else "human_resources" for label in labels if label.endswith(
        '_positive') or label.endswith('_neutral') or label.endswith('_negative')), None)
    sentiment = next(('positive' if label.endswith('_positive') else 'neutral' if label.endswith(
        '_neutral') else 'negative' for label in labels if aspect and label.startswith(aspect)), None)
    return aspect, sentiment

def get_result_view(labels, aspects):
    result = []
    for label in labels:
        aspect, sentiment = extract_aspect_sentiment([label])
        if aspect and sentiment:
            icon = aspects.get(aspect, "")
            formatted_aspect = aspect.replace("_", " ").title()
            formatted_label = f"{icon}   {formatted_aspect} {sentiment.capitalize()}"
            background_color = "#44475A"
            border_color = "#F8F8F2" if sentiment == "neutral" else "#FF5555" if sentiment == "negative" else "#50FA7B"
            styled_label = f'<div style="display: inline-block; padding: 5px 10px; margin: 5px; border-radius: 10px; background-color: {background_color}; border: 2px solid {border_color}; color: white; text-align: center; font-size: 16px;">{formatted_label}</div>'
            result.append(styled_label)
    return result

#! upload file
def extract_reviews(df):
    first_column_name = df.columns[1]
    reviews_list = df[first_column_name].tolist()

    return reviews_list

def count_labels(reviews, ensemble_model, labels):
    label_counter = Counter()

    for review in reviews:
        predicted_labels = predict(ensemble_model, review, labels)
        label_counter.update(predicted_labels)

    label_counts = dict(label_counter)

    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0

    return label_counts

def select_random_reviews(reviews, num_samples):
    if len(reviews) < num_samples:
        raise ValueError(
            "The number of samples requested exceeds the number of available reviews.")

    random_reviews = random.sample(reviews, num_samples)
    return random_reviews

def getBarViz(aspect_sentiments, colors):
    aspects = ['attractions', 'amenities', 'accessibility',
               'image', 'price', 'human_resources']
    sentiments = ['positive', 'neutral', 'negative']

    data = {aspect.replace('_', ' ').capitalize(): {sentiment: 0 for sentiment in sentiments}
            for aspect in aspects}

    for key, count in aspect_sentiments.items():
        aspect, sentiment = key.rsplit('_', 1)
        aspect = aspect.replace('_', ' ').capitalize()
        if aspect in data and sentiment in data[aspect]:
            data[aspect][sentiment] = count

    counts_positive = [data[aspect]['positive'] for aspect in data]
    counts_neutral = [data[aspect]['neutral'] for aspect in data]
    counts_negative = [data[aspect]['negative'] for aspect in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.7
    index = np.arange(len(data))

    bar1 = ax.barh(index, counts_positive, bar_width,
                   label='Positive', color=colors['positive'])
    bar2 = ax.barh(index, counts_neutral, bar_width,
                   left=counts_positive, label='Neutral', color=colors['neutral'])
    bar3 = ax.barh(index, counts_negative, bar_width, left=np.add(
        counts_positive, counts_neutral), label='Negative', color=colors['negative'])

    for i in range(len(index)):
        if counts_positive[i] > 0:
            ax.text(counts_positive[i] / 2, i, str(counts_positive[i]), ha='center', va='center', color='white')
        if counts_neutral[i] > 0:
            ax.text(counts_positive[i] + counts_neutral[i] / 2, i, str(counts_neutral[i]), ha='center', va='center', color='black')
        if counts_negative[i] > 0:
            ax.text(counts_positive[i] + counts_neutral[i] + counts_negative[i] / 2, i, str(counts_negative[i]), ha='center', va='center', color='white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Counts')
    ax.set_ylabel('Aspects')
    ax.set_title('Aspect Sentiment Counts from Reviews', fontsize=16, pad=20)
    ax.set_yticks(index)
    ax.set_yticklabels(data.keys())
    # ax.legend()

    # ax.grid(axis='x', linestyle='--', alpha=0.7)

    return fig, ax
    
def getDonutViz(sentiment_counts, colors):
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())

    colors_list = [colors[label.lower()] for label in labels]

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                      colors=colors_list, startangle=90, wedgeprops={'edgecolor': 'white'})

    centre_circle = plt.Circle((0, 0), 0.7, color='white', linewidth=0)
    ax.add_artist(centre_circle)

    ax.axis('equal')

    legend_patches = [mpatches.Patch(
        color=colors[label.lower()], label=label) for label in labels]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(
        0.5, -0.15), ncol=len(labels), fontsize='large')

    ax.set_title('Overall Sentiment Distribution', fontsize=16, pad=20)

    return fig, ax

def fig_to_array(fig):
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    return image[:, :, :3]
    
def convert_to_sentiment_counts(aspect_sentiments):
    sentiment_counts = {
        'Positive': sum(v for k, v in aspect_sentiments.items() if 'positive' in k),
        'Neutral': sum(v for k, v in aspect_sentiments.items() if 'neutral' in k),
        'Negative': sum(v for k, v in aspect_sentiments.items() if 'negative' in k)
    }
    return sentiment_counts

def getViz(aspect_sentiments, sentiment_counts, colors):
    fig1, ax1 = getBarViz(aspect_sentiments, colors)
    fig2, ax2 = getDonutViz(sentiment_counts, colors)

    fig1.canvas.draw()
    image1 = np.frombuffer(fig1.canvas.tostring_rgb(), dtype='uint8')
    image1 = image1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig1)

    fig2.canvas.draw()
    image2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8')
    image2 = image2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig2)
    
    fig, axs = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [5, 3]})
    axs[0].imshow(image1)
    axs[0].axis('off')
    axs[1].imshow(image2)
    axs[1].axis('off')
    
    plt.tight_layout()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)  # st.pyplot is deprecated
    st.pyplot(fig)

def main():
    html_temp = """
        <style>
        .tab {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
        }

        .tab button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 17px;
        }

        .tab button:hover {
            background-color: #ddd;
        }

        .tab button.active {
            background-color: #ccc;
        }

        .tabcontent {
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
        }
        </style>
        """

    html_temp = """
    <div>
    <h1 style="color:#8BE9FD;text-align:left;"> 🏝️ ABSA Tourism Review</h1>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.markdown(
        '<h1 style="color:#F8F8F2;">ℹ️ Project Description</h1>', unsafe_allow_html=True)

    st.sidebar.write("""
        <p>Welcome to our Aspect-based Sentiment Analysis tool for Borobudur and Prambanan temples!</p>
        <p>This project allows you to predict sentiment for six key tourism aspects, as recognized by the <strong>World Tourism Organization</strong> (<a href="https://www.unwto.org/global/publication/practical-guide-tourism-destination-management">WTO</a>): 
        <span style='color:#50FA7B'> Attractions</span>, 
        <span style='color:#50FA7B'> Amenity</span>, 
        <span style='color:#50FA7B'> Accessibility</span>, 
        <span style='color:#50FA7B'> Image</span>, 
        <span style='color:#50FA7B'> Price</span>, and 
        <span style='color:#50FA7B'> Human Resource</span>.
        <p>Understanding visitor e-WOM (Electronic Word of Mouth) is crucial for maintaining and improving the visitor experience at Borobudur and Prambanan temples. By predicting sentiments for different aspects, temple management can identify areas needing improvement to achieve sustainable tourism.</p>
        """, unsafe_allow_html=True)

    st.sidebar.markdown(
        '<h1 style="color:#F8F8F2;">💡How it works?</h1>', unsafe_allow_html=True)

    st.sidebar.write("""
        <p><strong style='color:#F1FA8C;'>Input Your Review</strong>: Enter your review sentence regarding Borobudur and Prambanan temples into the provided text box.</p>
        <p><strong style='color:#F1FA8C;'>Click Predict</strong>: Once your review is entered, click the <span style='color:#FFB86C;'>Predict button</span> to generate the aspect-based sentiment predictions.</p>
        <p><strong style='color:#F1FA8C;'>View Results</strong>: The aspect-based sentiment analysis results will be displayed, indicating whether the sentiment expressed in the review is positive, negative, or neutral for each aspect.</p>
        """, unsafe_allow_html=True)

    st.markdown('<h4 style="color:#BD93F9;">🔍 Analyze Aspect-Based Sentiments for Borobudur and Prambanan Temple Reviews</h4>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = ["Input Text", "Upload File"]

    tab_selected = st.selectbox("Ready to explore ABSA? Pick a task to start!", tabs)
    
    st.markdown("---")
    
    if tab_selected == "Input Text":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h5 style='color: #50FA7B;'>Input Text</h5>", unsafe_allow_html=True)
        
        user_input = st.text_area("Enter your sentence:")

        if st.button('Predict') or user_input:
            result = predict(ensemble_model, user_input, labels)
            result_views = get_result_view(result, aspects)
            for result_view in result_views:
                st.write(result_view, unsafe_allow_html=True)

    elif tab_selected == "Upload File":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h5 style='color: #50FA7B;'>Upload File</h5>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload a file", type=['csv', 'xls', 'xlsx'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                reviews = extract_reviews(df)
            
                if st.button('Analyze Data'):
                    aspect_sentiments = count_labels(reviews, ensemble_model, labels)
                    sentiment_counts = convert_to_sentiment_counts(
                        aspect_sentiments)
                    
                    st.markdown(f"**Total Reviews:** {len(reviews)}")
                    getViz(aspect_sentiments, sentiment_counts, colors)
    
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.warning(
        "Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")

    st.info(
        """
        Copyright © 2024 by TASI-2324-118

        Made with ❤️ by TASI-2324-118
        """
    )

if __name__ == '__main__':
    main()
