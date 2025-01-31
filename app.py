import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# ✅ Load sentiment & emotion models
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

# ✅ Sentiment Mapping (0=Negative, 1=Neutral, 2=Positive)
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ✅ Streamlit UI
st.title("Sentiment & Emotion Analysis Dashboard")

# ✅ Tabs for different features
tab1, tab2 = st.tabs(["Single Text Analysis", "Twitter Feed Analysis"])

# 🔹 Tab 1: Single Text Analysis
with tab1:
    st.subheader("Analyze a Single Text with Word-Level Sentiment")
    user_input = st.text_area("Enter text for analysis", "I love this product, but the service was terrible!")

    if st.button("Analyze"):
        if user_input:
            # ✅ Full Sentence Sentiment Analysis
            sentiment_result = sentiment_pipeline(user_input)[0]
            sentiment_label = sentiment_labels[int(sentiment_result["label"].split("_")[-1])]
            sentiment_score = round(sentiment_result["score"], 2)

            # ✅ Display full text sentiment
            st.subheader("Overall Sentiment Analysis")
            st.write(f"**Sentiment:** {sentiment_label} (Confidence: {sentiment_score})")

            # ✅ Word-Level Sentiment Analysis
            words = user_input.split()  # Tokenize text
            word_sentiments = [sentiment_pipeline(word)[0] for word in words]  # Get sentiment per word
            word_labels = [sentiment_labels[int(ws["label"].split("_")[-1])] for ws in word_sentiments]
            word_scores = [round(ws["score"], 2) for ws in word_sentiments]

            # ✅ Create DataFrame for visualization
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
    st.subheader("Analyze Sentiment of Twitter Feed")
    
    uploaded_file = st.file_uploader("Upload a CSV file with a column named 'text'", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # ✅ Check if 'text' column exists
        if "text" not in df.columns:
            st.error("⚠️ The uploaded CSV must contain a column named 'text'.")
        else:
            # ✅ Show sample data
            st.write("📌 Sample Data:", df.head())

            # ✅ Process Sentiment & Emotion Analysis (Avoid Redundant Calls)
            def analyze_sentiment(text):
                result = sentiment_pipeline(text)[0]
                return sentiment_labels[int(result["label"].split("_")[-1])], round(result["score"], 2)

            def analyze_emotion(text):
                result = emotion_pipeline(text)[0]
                return result["label"], round(result["score"], 2)

            # ✅ Apply analysis
            df[["Predicted Sentiment", "Sentiment Confidence"]] = df["text"].apply(lambda x: pd.Series(analyze_sentiment(x)))
            df[["Predicted Emotion", "Emotion Confidence"]] = df["text"].apply(lambda x: pd.Series(analyze_emotion(x)))

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

            # ✅ Evaluation: Accuracy & F1-Score (if ground truth exists)
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
