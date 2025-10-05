import streamlit as st
import asyncio

import ingest

api_key = st.write(st.secrets["API_KEY"])
import search_agent
import logs



# --- Initialization ---
@st.cache_resource
def init_agent():
    REPO_OWNER = "ultralytics"
    REPO_NAME = "ultralytics"

    def filter(doc):
        return "data-engineering" in doc["filename"]

    st.write("🔄 Indexing repo...")
    index, vindex = ingest.index_data(REPO_OWNER, REPO_NAME, vector=True)
    agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME, vindex=vindex)
    return agent

agent = init_agent()

# --- Streamlit UI ---
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Ultralytics Assistant")
st.caption("Ask me anything from the ultralytics repository")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Streaming helper ---
def stream_response(prompt: str):
    async def agen():
        try:
            async with agent.run_stream(user_prompt=prompt) as result:
                last_len = 0
                full_text = ""
                async for chunk in result.stream_output(debounce_by=0.01):
                    new_text = chunk[last_len:]
                    last_len = len(chunk)
                    full_text = chunk
                    if new_text:
                        yield new_text
                # log once complete
                logs.log_interaction_to_file(agent, result.new_messages())
                st.session_state._last_response = full_text
        except Exception as e:
            yield f"⚠️ Error during response generation: {e}"

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen_obj = agen()

    try:
        while True:
            piece = loop.run_until_complete(agen_obj.__anext__())
            yield piece
    except StopAsyncIteration:
        return
    except Exception as e:
        yield f"⚠️ Unexpected error: {e}"

# --- Chat input ---
prompt = st.chat_input("Ask your question...")

# Error handling for user input
MAX_INPUT_LENGTH = 1000  # adjust as needed

if prompt is not None:
    prompt = prompt.strip()
    if not prompt:
        st.warning("⚠️ Please enter a question before submitting.")
    elif len(prompt) > MAX_INPUT_LENGTH:
        st.warning(f"⚠️ Your question is too long (max {MAX_INPUT_LENGTH} characters).")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant message (streamed)
        with st.chat_message("assistant"):
            try:
                response_text = st.write_stream(stream_response(prompt))
            except Exception as e:
                response_text = f"⚠️ Error: {e}"
                st.error(response_text)

        # Save full response to history
        final_text = getattr(st.session_state, "_last_response", response_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})