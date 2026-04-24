import streamlit as st
import preprocessor
import helper
import ml_reply_model
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="WhatsApp ML Analytics Dashboard", layout="wide")

# ================= SESSION STATE =================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ================= STYLE =================
st.markdown("""
<style>
.main {background-color: #0E1117;}
.block-container {padding-top: 2rem;}
.metric-card {
    background: linear-gradient(145deg, #1E222B, #161A22);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: #9DA5B4;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("📊 WhatsApp Analytics")
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat File")

if uploaded_file is not None:

    data = uploaded_file.getvalue().decode("utf-8")
    df = preprocessor.preprocess(data)
    df = df[df['user'] != 'group_notification']

    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Select User", user_list)

    if st.sidebar.button("🚀 Analyze"):
        st.session_state.analyzed = True

    if st.session_state.analyzed:

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview",
            "📅 Timeline",
            "📈 Activity",
            "😊 Sentiment",
            "🔥 Abuse",
            "🧠 Reply Prediction"
        ])

        # ================= OVERVIEW =================
        with tab1:
            num_messages, words, media, links = helper.fetch_stats(selected_user, df)
            avg_length = round(words / num_messages, 2) if num_messages > 0 else 0

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Messages", num_messages)
            col2.metric("Words", words)
            col3.metric("Media", media)
            col4.metric("Links", links)
            col5.metric("Avg Msg Length", avg_length)

        # ================= TIMELINE =================
        with tab2:
            timeline = helper.monthly_timeline(selected_user, df)
            fig = px.line(timeline, x="time", y="message", markers=True)
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        # ================= ACTIVITY =================
        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                busy_day = helper.week_activity_map(selected_user, df)
                day_df = busy_day.reset_index()
                day_df.columns = ["Day", "Messages"]
                fig1 = px.bar(day_df, x="Day", y="Messages", color="Day")
                fig1.update_layout(template="plotly_dark")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                busy_month = helper.month_activity_map(selected_user, df)
                month_df = busy_month.reset_index()
                month_df.columns = ["Month", "Messages"]
                fig2 = px.bar(month_df, x="Month", y="Messages", color="Month")
                fig2.update_layout(template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)

        # ================= SENTIMENT =================
        with tab4:
            sentiment_count, _ = helper.sentiment_analysis(selected_user, df)
            sentiment_df = sentiment_count.reset_index()
            sentiment_df.columns = ["Sentiment", "Count"]

            fig = px.pie(sentiment_df, names="Sentiment", values="Count")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        # ================= ABUSE =================
        with tab5:
            total_abusive, abuse_percentage, most_abusive_user, _, abusive_word_df, _ = helper.abuse_analysis(selected_user, df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Abusive Words", total_abusive)
            col2.metric("Toxicity %", abuse_percentage)
            col3.metric("Most Abusive User", most_abusive_user)

            if not abusive_word_df.empty:
                fig = px.bar(abusive_word_df, x="Count", y="Abusive Word", orientation="h")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("📄 User-wise Abusive Word Table")

            abuse_table = helper.abusive_word_user_table(df)

            if not abuse_table.empty:
                st.dataframe(abuse_table)

                csv = abuse_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download CSV",
                    data=csv,
                    file_name="abusive_word_usage.csv",
                    mime="text/csv"
                )
            else:
                st.info("No abusive words found.")

        # ================= ML REPLY PREDICTION =================
        with tab6:

            if st.button("Train Reply Model"):
                metrics = ml_reply_model.train_reply_model(df)

                if "Error" in metrics:
                    st.error(metrics["Error"])
                else:
                    st.success("Model Trained Successfully!")
                    for model_name, values in metrics.items():
                        st.markdown(f"### {model_name}")
                        st.write(f"MAE: {values['MAE']:.2f}")
                        st.write(f"RMSE: {values['RMSE']:.2f}")
                        st.write(f"R² Score: {values['R2']:.2f}")

            st.divider()

            word_count = st.number_input("Message Word Count", 1, 200, 10)
            hour = st.slider("Hour of Message", 0, 23, 12)
            user = st.selectbox("Sender", df["user"].unique())
            day = st.selectbox("Day", df["day_name"].unique())

            if st.button("Predict Reply Time"):
                result = ml_reply_model.predict_reply_time(
                    word_count, hour, user, day
                )

                if "Error" in result:
                    st.warning(result["Error"])
                else:
                    st.success(f"Predicted Reply Time: {result['Prediction']} minutes")
