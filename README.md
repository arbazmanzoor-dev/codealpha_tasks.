# VoltCart FAQ Chatbot

A retrieval-based FAQ chatbot for a sample online electronics store. It answers questions about orders, shipping, returns, payments, warranty and accounts using classic NLP (NLTK preprocessing, TF-IDF vectors and cosine similarity) instead of a large language model.

## Project structure

| File | Purpose |
|---|---|
| `faqs.json` | 34 FAQs, each with a `question`, `answer`, `category` and optional `variants` (other phrasings of the question) |
| `preprocess.py` | NLTK text pipeline, which downloads the NLTK data it needs on first run |
| `chatbot.py` | `FAQChatbot` engine (intent rules, TF-IDF matching, fallback) plus the `--cli` mode |
| `app.py` | Streamlit chat UI |
| `test_chatbot.py` | Runs 10 sample queries and prints the match and score for each |

## Approach

1. **Preprocessing** (`preprocess.py`), applied the same way to the FAQ questions and the user's query:
   lowercase → remove punctuation → tokenize (`word_tokenize`) → remove stopwords (NLTK English list, keeping negations such as *not*) → POS-aware lemmatization (`WordNetLemmatizer`, so *charged* → *charge* and *arrived* → *arrive*) → map a small set of domain synonyms to one term (*parcel* → *package*, *overseas* → *internationally*, *money* → *refund*).
   Example: `"I've been charged TWICE for the same order!!"` → `charge twice order`.

2. **Vectorization.** A scikit-learn `TfidfVectorizer` is fitted on the preprocessed FAQ questions and their `variants`, using unigrams and bigrams with sublinear TF. TF-IDF gives a word a high weight when it is frequent in one question and rare across the rest, so distinctive words like *warranty* or *refund* count for more than common ones like *order*.

3. **Matching.** The query is transformed with the same vectorizer, and cosine similarity is computed against every question and variant vector. An FAQ's score is the best of its question and variants:

   `cos(q, d) = (q · d) / (‖q‖ ‖d‖)`, which gives 0 when the query and question share no terms and 1 when their term weights are identical.

   - If the best score is at least the **threshold** (0.3 by default), the bot returns that FAQ's answer.
   - Otherwise it gives a friendly fallback and suggests the **3 closest questions**. If no FAQ shares any word with the query, it suggests popular questions instead.

4. **Intents.** Greetings (*hi*, *good morning*), thanks and goodbyes (*bye*, *see you*) are caught with regular expressions before matching.

**Limitations.** TF-IDF only matches words, not meaning. A paraphrase that shares no content words with any FAQ can't be matched. For example, "where is my parcel?" reduces to just `package`, which has no words in common with "How can I track my order?". Two things work around this:
- the synonym map, and
- `variants`: extra phrasings such as "Where is my package?" listed on an FAQ.

When you see a real question the bot misses, add it as a variant. Changing a variant can shift which FAQ wins for other questions, so rerun `python test_chatbot.py` afterwards. Sentence embeddings, such as `sentence-transformers`, would remove the need for hand-written variants.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Web UI**
```bash
streamlit run app.py
```
Turn on **Debug mode** in the sidebar to see the matched question, its category and the similarity score under each reply. You can also adjust the threshold there.

**Command line**
```bash
python chatbot.py --cli            # add --debug to show scores, --threshold 0.4 to change the cutoff
```

**Sample test**
```bash
python test_chatbot.py
```

The first run downloads the NLTK data (punkt, stopwords, wordnet and the POS tagger) automatically.
