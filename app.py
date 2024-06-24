import streamlit as st
from model import check

def check_new(text):
    return text
def main():
    st.title("Text Classifier App")

    # Input text
    text_input = st.text_area("Enter text to classify:", "")

    if st.button("Classify"):
        if text_input:
            label = check(text_input)
            print(label)
            if label == 1:
                st.write(f"The message is spam")
            else:
                st.write(f"The message is ham")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
