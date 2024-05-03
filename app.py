import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt

pipe_lr=joblib.load(open('models/emotion_classifier_pipe_lr_03_jan_2022.pkl', 'rb'))

#function to read the emotion
def predict_emotions(docx):
    results=pipe_lr.predict([docx] )
    return results

def get_prediction_proba(docx):
    results=pipe_lr.predict_proba([docx] )
    return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    st.title('SoulSakhi:  Your Partner in Emotional Wellness')

    st.subheader("Feel Free Zone: Express Yourself")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Please enter your text")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        col1,col2 = st.columns(2)
        prediction=predict_emotions(raw_text)
        probability=get_prediction_proba(raw_text)
        with col1:
            st.success('Your Input')
            st.write(raw_text)

            st.success("Results")
            emoji_icon= emotions_emoji_dict[prediction[0]]
            st.write("{}:{}".format(prediction[0],emoji_icon))
            st.write("Confidence: {}".format(np.max(probability)))

        with col2:
            st.success('Prediction Probability')
            st.write(probability)
            proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
            st.write(proba_df.transpose())
            proba_df_clean=proba_df.transpose().reset_index()
            proba_df_clean.columns=["emotions","probability"]

        fig=alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability',color='emotions')
        st.altair_chart(fig,use_container_width=True)

if __name__ == "__main__":
    main()
