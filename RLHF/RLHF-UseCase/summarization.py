from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st

# Constants
MAX_TARGET_LENGTH = 256
MIN_TARGET_LENGTH = 5
LEARNING_RATE = 5e-4
BATCH_SIZE = 1

# Load the saved model and tokenizer
SAVE_PATH = "/Users/srikarkilari/InternshipDemo/summarization_files"
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(SAVE_PATH)
summarization_pipeline = pipeline('summarization', model=model, tokenizer=tokenizer, framework="pt")

def fine_tune_model_on_single_example(model, input_text, corrected_summary, tokenizer, epochs=3, clip_value=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    inputs = tokenizer([input_text], padding=True, truncation=True, return_tensors="pt")
    targets = tokenizer([corrected_summary], padding=True, truncation=True, return_tensors="pt")

    inputs = inputs.input_ids.to(device)
    targets = targets.input_ids.to(device)

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 

    for epoch in range(epochs):
        for input_batch, target_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=input_batch, labels=target_batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  
            optimizer.step()

# Initialize session states
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'input_text' not in st.session_state:
    st.session_state.input_text = ''
if 'initial_summary' not in st.session_state:
    st.session_state.initial_summary = ''
if 'corrected_summary' not in st.session_state:
    st.session_state.corrected_summary = ''

st.title('Text Summarization Chatbot')

if st.session_state.step == 1:
    st.session_state.input_text = st.text_area("Enter text to summarize (or type 'BYE' to exit):", st.session_state.input_text)
    if st.session_state.input_text.strip().upper() == 'BYE':
        st.write("Thank you! Goodbye!")
        st.session_state.step = 4
    elif st.button('Summarize'):
        st.write(f"**Provided Text:**\n{st.session_state.input_text}\n")  # Displaying the provided input text
        st.session_state.initial_summary = summarization_pipeline(st.session_state.input_text, max_length=MAX_TARGET_LENGTH, min_length=MIN_TARGET_LENGTH)[0]['summary_text']
        st.session_state.step = 2

elif st.session_state.step == 2:
    st.write(f"**Provided Text:**\n{st.session_state.input_text}\n")  # Displaying the provided input text
    st.write(f"**Initial Summary:** {st.session_state.initial_summary}")
    st.session_state.corrected_summary = st.text_area("Please provide a corrected summary (if needed):", st.session_state.corrected_summary)
    
    if st.button('Fine-tune and Regenerate Summary'):
        if st.session_state.corrected_summary:
            fine_tune_model_on_single_example(model, st.session_state.input_text, st.session_state.corrected_summary, tokenizer)
            refined_summary = summarization_pipeline(st.session_state.input_text, max_length=MAX_TARGET_LENGTH, min_length=MIN_TARGET_LENGTH)[0]['summary_text']
            st.write(f"**Refined Summary (based on feedback):** {refined_summary}")
            st.session_state.step = 3
        else:
            st.write("No corrected summary provided. Using initial summary.")
            st.session_state.step = 3

elif st.session_state.step == 3:
    if st.button('Continue with a new text'):
        st.session_state.step = 1
        st.session_state.input_text = ''
        st.session_state.initial_summary = ''
        st.session_state.corrected_summary = ''
