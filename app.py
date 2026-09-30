"""Streamlit chat UI for the FAQ chatbot. Run with: streamlit run app.py"""

import streamlit as st

from chatbot import DEFAULT_THRESHOLD, FAQChatbot

st.set_page_config(page_title="VoltCart Help", page_icon="💬")


@st.cache_resource
def load_bot() -> FAQChatbot:
    return FAQChatbot()


bot = load_bot()

with st.sidebar:
    st.header("Settings")
    debug = st.toggle("Debug mode", help="Show the matched FAQ question and its similarity score")
    bot.threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05)
    if st.button("Clear conversation"):
        st.session_state.messages = []
    st.caption(f"{len(bot.faqs)} FAQs loaded · TF-IDF + cosine similarity")

st.title("💬 VoltCart Help Center")
st.caption("Ask about orders, shipping, returns, payments, warranty or your account.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_debug(info: dict) -> None:
    if info.get("score") is None:
        st.caption(f"🛠 intent: `{info['intent']}` (rule-based)")
        return
    if info.get("matched_question") is None:
        st.caption(f"🛠 intent: `{info['intent']}` · no FAQ shares any terms with this query (similarity 0)")
        return
    st.caption(
        f"🛠 intent: `{info['intent']}` · best match: *{info['matched_question']}* "
        f"({info['category']}) · similarity: **{info['score']:.3f}** · threshold: {info['threshold']:.2f}"
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=msg["role"] == "assistant")
        if debug and msg.get("debug"):
            render_debug(msg["debug"])

if prompt := st.chat_input("Type your question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    r = bot.respond(prompt)
    info = {"intent": r.intent, "matched_question": r.matched_question, "category": r.category,
            "score": r.score, "threshold": bot.threshold}
    content = r.text
    if r.intent == "faq":
        content = f"{r.text}\n\n<sub>📂 {r.category}</sub>"
    elif r.intent == "fallback" and r.suggestions:
        content = r.text.split("\n")[0] + "\n" + "\n".join(f"- {q}" for q, _ in r.suggestions)

    st.session_state.messages.append({"role": "assistant", "content": content, "debug": info})
    with st.chat_message("assistant"):
        st.markdown(content, unsafe_allow_html=True)
        if debug:
            render_debug(info)
