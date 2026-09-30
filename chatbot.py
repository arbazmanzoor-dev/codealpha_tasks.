"""FAQ chatbot engine: TF-IDF + cosine similarity retrieval over faqs.json.

Run `python chatbot.py --cli` for an interactive command-line chat.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess

FAQ_PATH = Path(__file__).with_name("faqs.json")
DEFAULT_THRESHOLD = 0.3

GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|hiya|howdy|greetings|good\s+(morning|afternoon|evening)|yo)\b[\s!.,]*(there|bot)?[\s!.]*$", re.I)
GOODBYE_RE = re.compile(r"^\s*(bye+|goodbye|good\s*bye|see\s+(you|ya)|farewell|exit|quit|that'?s\s+all|cya|take\s+care)\b.*$", re.I)
THANKS_RE = re.compile(r"^\s*(thanks|thank\s+you|thx|ty|cheers|much\s+appreciated)\b[\s\w!.,]*$", re.I)

GREETINGS = [
    "Hello! 👋 I'm the VoltCart help bot. Ask me about orders, shipping, returns, payments, warranty or your account.",
    "Hi there! How can I help you today? You can ask me things like \"How do I track my order?\"",
]
GOODBYES = ["Goodbye! Thanks for shopping with VoltCart. 👋", "Bye! Have a great day, and come back any time you have a question."]
THANKS = ["You're welcome! Anything else I can help with?", "Happy to help! Let me know if you have another question."]
POPULAR_QUESTIONS = ["How can I track my order?", "What is your return policy?", "What payment methods do you accept?"]
FALLBACK = "Sorry, I'm not sure I understood that. 🤔 Could you rephrase it? Here are some questions I can answer that might be related:"


@dataclass
class Response:
    """What the bot says, plus debug information about how it decided."""

    text: str
    intent: str  # "greeting" | "goodbye" | "thanks" | "faq" | "fallback"
    matched_question: str | None = None
    category: str | None = None
    score: float | None = None
    suggestions: list[tuple[str, float]] = field(default_factory=list)


class FAQChatbot:
    def __init__(self, faq_path: str | Path = FAQ_PATH, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        with open(faq_path, encoding="utf-8") as f:
            self.faqs: list[dict] = json.load(f)
        self.questions = [faq["question"] for faq in self.faqs]

        # Each FAQ is indexed by its question plus any optional "variants"
        # (other phrasings); doc_faq maps each indexed text back to its FAQ.
        docs, self.doc_faq = [], []
        for i, faq in enumerate(self.faqs):
            for text in [faq["question"], *faq.get("variants", [])]:
                docs.append(text)
                self.doc_faq.append(i)

        # Preprocessing (lowercase, punctuation, stopwords, lemmatization) is done
        # by preprocess(); the vectorizer only splits on whitespace and weights terms.
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess,
            token_pattern=r"\S+",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.faq_matrix = self.vectorizer.fit_transform(docs)

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Return (faq index, cosine similarity) for every FAQ, best first.

        An FAQ's score is the best score among its question and variants.
        """
        query_vec = self.vectorizer.transform([query])
        doc_scores = cosine_similarity(query_vec, self.faq_matrix).ravel()
        scores = [0.0] * len(self.faqs)
        for faq_idx, score in zip(self.doc_faq, doc_scores):
            scores[faq_idx] = max(scores[faq_idx], float(score))
        return sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

    def respond(self, message: str) -> Response:
        message = message.strip()
        if not message:
            return Response("Please type a question. 🙂", intent="fallback")
        if GREETING_RE.match(message):
            return Response(random.choice(GREETINGS), intent="greeting")
        if GOODBYE_RE.match(message):
            return Response(random.choice(GOODBYES), intent="goodbye")
        if THANKS_RE.match(message):
            return Response(random.choice(THANKS), intent="thanks")

        ranked = self.rank(message)
        best_idx, best_score = ranked[0]
        best = self.faqs[best_idx]
        if best_score >= self.threshold:
            return Response(best["answer"], intent="faq", matched_question=best["question"],
                            category=best["category"], score=best_score)

        if best_score > 0:
            suggestions = [(self.questions[i], s) for i, s in ranked[:3] if s > 0]
        else:  # no word overlap at all: offer popular questions instead of arbitrary ones
            suggestions = [(q, 0.0) for q in POPULAR_QUESTIONS]
        text = FALLBACK + "\n" + "\n".join(f"  • {q}" for q, _ in suggestions)
        has_match = best_score > 0
        return Response(text, intent="fallback", matched_question=best["question"] if has_match else None,
                        category=best["category"] if has_match else None, score=best_score, suggestions=suggestions)


def run_cli(bot: FAQChatbot, debug: bool) -> None:
    print("VoltCart FAQ bot — type 'bye' to exit.\n")
    while True:
        try:
            message = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        response = bot.respond(message)
        print(f"Bot: {response.text}")
        if debug and response.score is not None:
            print(f"     [debug] intent={response.intent} score={response.score:.3f} matched={response.matched_question!r}")
        print()
        if response.intent == "goodbye":
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="FAQ chatbot (TF-IDF + cosine similarity)")
    parser.add_argument("--cli", action="store_true", help="chat in the terminal")
    parser.add_argument("--debug", action="store_true", help="show the matched question and similarity score")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="minimum similarity to answer (default 0.3)")
    args = parser.parse_args()

    if not args.cli:
        parser.print_help()
        print("\nTip: run `python chatbot.py --cli` for terminal chat, or `streamlit run app.py` for the web UI.")
        return
    run_cli(FAQChatbot(threshold=args.threshold), debug=args.debug)


if __name__ == "__main__":
    main()
