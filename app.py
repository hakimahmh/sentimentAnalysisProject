import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score

# ✅ Load sentiment & emotion models
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# ✅ Sentiment Mapping (0=Negative, 1=Neutral, 2=Positive)
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ✅ Streamlit UI
st.title("📊 Sentiment & Emotion Analysis Dashboard")

# ✅ Tabs for different features
tab1, tab2 = st.tabs(["Single Text Analysis", "Social Media Feed Analysis"])

# 🔹 Tab 1: Single Text Analysis
with tab1:
    st.subheader("Analyze a Single Text for Sentiment & Emotion")
    user_input = st.text_area("Enter text for analysis", "I love this product, but the service was terrible!")

    if st.button("Analyze"):
        if user_input:
            # ✅ Full Sentence Sentiment Analysis
            sentiment_result = sentiment_pipeline(user_input)[0]
            sentiment_label = sentiment_labels[int(sentiment_result["label"].split("_")[-1])]
            sentiment_score = round(sentiment_result["score"], 2)

            # ✅ Emotion Analysis
            emotion_result = emotion_pipeline(user_input)[0]
            emotion_label = emotion_result["label"]
            emotion_score = round(emotion_result["score"], 2)

            # ✅ Display results
            st.subheader("Overall Sentiment & Emotion Analysis")
            st.write(f"**Sentiment:** {sentiment_label} (Confidence: {sentiment_score})")
            st.write(f"**Emotion:** {emotion_label} (Confidence: {emotion_score})")

            # ✅ Word-Level Sentiment Analysis
            words = user_input.split()
            word_sentiments = [sentiment_pipeline(word)[0] for word in words]
            word_labels = [sentiment_labels[int(ws["label"].split("_")[-1])] for ws in word_sentiments]
            word_scores = [round(ws["score"], 2) for ws in word_sentiments]

            # ✅ DataFrame for visualization
            word_df = pd.DataFrame({"Word": words, "Sentiment": word_labels, "Score": word_scores})

            # ✅ Display table
            st.subheader("Word-Level Sentiment Breakdown")
            st.write(word_df)

            # ✅ Visualization: Word-Level Sentiment Scores
            st.subheader("📊 Word-Level Sentiment Scores")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=word_df["Word"], y=word_df["Score"], hue=word_df["Sentiment"], palette="viridis", ax=ax)
            ax.set_title("Sentiment Confidence per Word")
            ax.set_ylabel("Confidence Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Please enter some text to analyze!")

# 🔹 Tab 2: Twitter Feed Analysis
with tab2:
    st.subheader("Analyze Sentiment & Emotion in Social Media Feed Using Dataset")

    uploaded_file = st.file_uploader("Upload a CSV file with a column named 'text'", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # ✅ Check if 'text' column exists
        if "text" not in df.columns:
            st.error("⚠️ The uploaded CSV must contain a column named 'text'.")
        else:
            # ✅ Show sample data
            st.write("📌 Sample Data:", df.head())

            # ✅ Define sentiment analysis function
            def analyze_sentiment(text):
                result = sentiment_pipeline(text)[0]
                return sentiment_labels[int(result["label"].split("_")[-1])], round(result["score"], 2)

            # ✅ Define emotion analysis function
            def analyze_emotion(text):
                if text:  # Check if text is not empty
                    result = emotion_pipeline(text)[0]
                    return result["label"], round(result["score"], 2)
                else:
                    return "No Emotion", 0.0

            # ✅ Apply sentiment and emotion analysis
            df[["Predicted Sentiment", "Sentiment Confidence"]] = df["text"].apply(lambda x: pd.Series(analyze_sentiment(x)))
            df[["Predicted Emotion", "Emotion Confidence"]] = df["text"].apply(lambda x: pd.Series(analyze_emotion(x)))

            # ✅ Set ground truth from predicted sentiment if missing
            if "true_sentiment" not in df.columns:
                df["true_sentiment"] = df["Predicted Sentiment"]

            # ✅ Display Processed Results
            st.subheader("📊 Results")
            st.write(df.head(10))

            # ✅ Visualization: Sentiment Distribution
            st.subheader("📈 Sentiment Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=df["Predicted Sentiment"], palette="viridis", ax=ax)
            ax.set_title("Sentiment Analysis Distribution")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # ✅ Visualization: Emotion Distribution
            st.subheader("📈 Emotion Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=df["Predicted Emotion"], palette="coolwarm", ax=ax)
            ax.set_title("Emotion Analysis Distribution")
            ax.set_xlabel("Emotion")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ✅ Model Evaluation (if ground truth exists)
            if "true_sentiment" in df.columns:
                st.subheader("📊 Model Evaluation")

                # ✅ Convert labels to match model output
                label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
                df["true_sentiment"] = df["true_sentiment"].map(label_mapping)
                df["Predicted Sentiment"] = df["Predicted Sentiment"].map(label_mapping)

                # ✅ Compute Accuracy & F1-Score
                accuracy = accuracy_score(df["true_sentiment"], df["Predicted Sentiment"])
                f1 = f1_score(df["true_sentiment"], df["Predicted Sentiment"], average="weighted")

                # ✅ Display metrics
                st.write(f"✅ **Accuracy:** {accuracy:.2f}")
                st.write(f"✅ **F1 Score:** {f1:.2f}")

            # ✅ Allow user to download processed data
            st.download_button(
                label="📥 Download Processed Data",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="twitter_sentiment_analysis.csv",
                mime="text/csv"
            )

            # ✅ Confidence-Based Evaluation
            if "Sentiment Confidence" in df.columns:
                st.subheader("📊 Confidence-Based Evaluation")

                avg_confidence = df["Sentiment Confidence"].mean()
                confidence_threshold = 0.7
                confident_predictions = df[df["Sentiment Confidence"] >= confidence_threshold]
                confidence_accuracy = len(confident_predictions) / len(df)

                # ✅ Display metrics
                st.write(f"✅ **Average Confidence Score:** {avg_confidence:.2f}")
                st.write(f"✅ **Confidence-Weighted Accuracy (Threshold {confidence_threshold}):** {confidence_accuracy:.2f}")

    else:
        st.warning("⚠️ Please upload a dataset.")
