import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from google_play_scraper import reviews

MODEL_PATH = "leon6523/t5-small-finetuned-opinosis"

@st.cache(allow_output_mutation=True)
def load_model(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    return tokenizer, model


def summarize_text(text: str, tokenizer, model, max_input_length: int = 512, max_summary_length: int = 128):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    )
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_summary_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    st.title("App Review Summarizer")
    st.write("Use a fine-tuned T5 model to summarize app reviews.")

    tokenizer, model = load_model(MODEL_PATH)

    mode = st.radio(
        "Choose input mode:",
        ("Manual Input", "Fetch from Google Play")
    )

    if mode == "Manual Input":
        review_text = st.text_area(
            "Enter the review text to summarize:",
            height=300
        )
        if st.button("Summarize"): 
            if not review_text.strip():
                st.error("Please provide some text to summarize.")
            else:
                summary = summarize_text(review_text, tokenizer, model)
                st.subheader("Summary")
                st.write(summary)

    else:
        package_name = st.text_input(
            "App package name (e.g. com.spotify.music):",
            value="com.spotify.music"
        )
        num = st.slider(
            "Number of reviews to summarize (each >=50 words):",
            min_value=10,
            max_value=50,
            value=10
        )
        if st.button("Fetch & Summarize"):
            if not package_name.strip():
                st.error("Please enter a valid package name.")
                return

            filtered = []
            token = None
            batch = max(num, 10)  # fetch at least 10 per batch
            while len(filtered) < num:
                result, token = reviews(
                    package_name,
                    lang='en',
                    country='us',
                    count=batch,
                    continuation_token=token
                )
                if not result:
                    break
                for r in result:
                    if len(r['content'].split()) >= 50:
                        filtered.append(r)
                    if len(filtered) >= num:
                        break
                if not token:
                    break

            if not filtered:
                st.warning("No reviews with at least 50 words were found.")
                return
            if len(filtered) < num:
                st.warning(
                    f"Only found {len(filtered)} reviews with >=50 words. Proceeding with available reviews."
                )

            # Show fetched reviews
            st.subheader(f"Fetched Reviews (>=50 words) - {len(filtered)} items")
            for i, r in enumerate(filtered, start=1):
                st.markdown(f"**{i}.** {r['content']}")

            # Summarize each filtered review
            st.subheader("Individual Summaries")
            summaries = [summarize_text(r['content'], tokenizer, model) for r in filtered]
            for s in summaries:
                st.write(f"- {s}")

            # Combined summary
            combined = " ".join(summaries)
            st.subheader("Combined Summary")
            final = summarize_text(combined, tokenizer, model)
            st.write(final)

if __name__ == "__main__":
    main()
