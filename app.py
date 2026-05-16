import streamlit as st
import tensorflow as tf

from model import get_resources, generate_text


def main():
    st.set_page_config(page_title="Next Words Prediction (LSTM)", layout="centered")
    st.title("Next Words Prediction (LSTM)")
    st.write("Enter a seed text and generate the next words using your trained LSTM model.")

    model, tokenizer, max_len, index_to_word = get_resources()

    with st.sidebar:
        st.header("Settings")
        seed_text = st.text_area("Seed text", value="are you a", height=120)
        n_words = st.slider(
            "Number of words to generate", min_value=1, max_value=50, value=10, step=1
        )
        generate_btn = st.button("Generate")

    if generate_btn:
        if not seed_text.strip():
            st.warning("Please enter seed text.")
            st.stop()

        with st.spinner("Generating text..."):
            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                index_to_word=index_to_word,
                seed_text=seed_text,
                max_len=max_len,
                n_words=n_words,
            )

        st.subheader("Generated text")
        st.write(result)

        continuation = result[len(seed_text) :].strip() if result.startswith(seed_text) else ""
        if continuation:
            st.caption("Continuation: " + continuation)


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()

