import numpy as np
import streamlit as st
import pickle

st.set_page_config(page_title="ABSA Tourism Review", page_icon="🏝️",
                   layout='centered', initial_sidebar_state="expanded")

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

html_temp = """
    <div>
    <h1 style="color:#8BE9FD;text-align:left;"> 🏝️ ABSA Tourism Review</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.markdown('<h1 style="color:#F8F8F2;">ℹ️ Project Description</h1>', unsafe_allow_html=True)

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

st.sidebar.markdown('<h1 style="color:#F8F8F2;">💡How it works?</h1>', unsafe_allow_html=True)

st.sidebar.write("""
    <p><strong style='color:#F1FA8C;'>Input Your Review</strong>: Enter your review sentence regarding Borobudur and Prambanan temples into the provided text box.</p>
    <p><strong style='color:#F1FA8C;'>Click Predict</strong>: Once your review is entered, click the <span style='color:#FFB86C;'>Predict button</span> to generate the aspect-based sentiment predictions.</p>
    <p><strong style='color:#F1FA8C;'>View Results</strong>: The aspect-based sentiment analysis results will be displayed, indicating whether the sentiment expressed in the review is positive, negative, or neutral for each aspect.</p>
    """, unsafe_allow_html=True)

st.markdown('<h3 style="color:#BD93F9;">🔍 Analyze Aspect-Based Sentiments for Borobudur and Prambanan Temple Reviews</h3>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
user_input = st.text_area("Enter your sentence:")

def extract_aspect_sentiment(labels):
    aspect = next((label.split('_')[0] if label.split('_')[0] != "human" else "human_resources" for label in labels if label.endswith(
        '_positive') or label.endswith('_neutral') or label.endswith('_negative')), None)
    sentiment = next(('positive' if label.endswith('_positive') else 'neutral' if label.endswith(
        '_neutral') else 'negative' for label in labels if aspect and label.startswith(aspect)), None)
    return aspect, sentiment

aspects = {
    "attractions": "🔮",
    "amenities": "🛎️",
    "accessibility": "🏄‍♂️",
    "image": "🖼️",
    "price": "💰",
    "human_resources": "👨‍💼"
}

def getResultView(labels, aspects):
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

if st.button('Predict') or user_input:
    result = predict(ensemble_model, user_input, labels)
    resultViews = getResultView(result, aspects)
    for resultView in resultViews:
        st.write(resultView, unsafe_allow_html=True)
        
st.markdown("<br><br><br>", unsafe_allow_html=True)
        
st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")

st.info(
    """
    Copyright © 2024 by TASI-2324-118

    Made with ❤️ by TASI-2324-118
    """
)