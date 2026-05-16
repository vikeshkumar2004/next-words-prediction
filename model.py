import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def _load_artifacts():
    """Load tokenizer, max_len, and LSTM model from the current directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "lstm_model (1).h5")
    tokenizer_path = os.path.join(base_dir, "tokenizer.pkl")
    max_len_path = os.path.join(base_dir, "max_len.pkl")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(max_len_path, "rb") as f:
        max_len = pickle.load(f)

    model = load_model(model_path)

    # Build reverse mapping.
    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    return model, tokenizer, max_len, index_to_word


@st.cache_resource(show_spinner=True)
def get_resources():
    return _load_artifacts()


def next_word_predict(model, tokenizer, index_to_word, text, max_len):
    text = (text or "").lower().strip()
    if not text:
        return ""

    seq = tokenizer.texts_to_sequences([text])[0]
    if len(seq) == 0:
        return ""

    seq = pad_sequences([seq], maxlen=max_len, padding="pre")

    pred = model.predict(seq, verbose=0)
    pred_index = int(np.argmax(pred, axis=-1)[0] if pred.ndim == 2 else np.argmax(pred))

    return index_to_word.get(pred_index, "")


def generate_text(model, tokenizer, index_to_word, seed_text, max_len, n_words):
    seed_text = (seed_text or "").strip()
    if not seed_text:
        return ""

    generated = seed_text
    for _ in range(int(n_words)):
        nxt = next_word_predict(model, tokenizer, index_to_word, generated, max_len)
        if not nxt:
            break
        generated = generated + " " + nxt

    return generated


def main():
    st.set_page_config(page_title="Next Words Prediction (LSTM)", layout="centered")
    st.title("Next Words Prediction (LSTM)")
    st.write("Enter a seed text and generate the next words using your trained LSTM model.")

    model, tokenizer, max_len, index_to_word = get_resources()

    with st.sidebar:
        st.header("Settings")
        seed_text = st.text_area("Seed text", value="are you a", height=120)
        n_words = st.slider("Number of words to generate", min_value=1, max_value=50, value=10, step=1)
        generate_btn = st.button("Generate")

    if generate_btn:
        if not seed_text.strip():
            st.warning("Please enter seed text.")
            st.stop()

        with st.spinner("Generating text..."):
            result = generate_text(model, tokenizer, index_to_word, seed_text, max_len, n_words)

        st.subheader("Generated text")
        st.write(result)

        # Also show just the continuation
        continuation = result[len(seed_text):].strip() if result.startswith(seed_text) else ""
        if continuation:
            st.caption("Continuation: " + continuation)


if __name__ == "__main__":
    # Helps streamlit reruns behave more deterministically.
    tf.get_logger().setLevel("ERROR")
    main()

