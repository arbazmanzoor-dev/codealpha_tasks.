"""Run the bot against 10 sample questions and print what it matched."""

from chatbot import FAQChatbot

SAMPLES = [
    # (query, question we expect it to match, or None for greeting/fallback)
    ("How can I track my order?", "How can I track my order?"),                                   # exact
    ("where is my parcel right now", "How can I track my order?"),                                 # paraphrase
    ("Do you deliver overseas?", "Do you ship internationally?"),                                  # paraphrase + synonyms
    ("what's the deal if the laptop shows up broken", "What if my product arrived damaged or defective?"),
    ("i forgot my login password", "I forgot my password. How do I reset it?"),
    ("Can I pay in monthly installments?", "Do you offer EMI or buy now pay later options?"),
    ("my card got charged 2 times for one order", "I was charged twice for the same order. What should I do?"),
    ("how long until I get my money back", "How long does it take to get a refund?"),
    ("Hello!", None),                                                                              # greeting
    ("What's the weather like in Paris tomorrow?", None),                                          # unrelated -> fallback
]


def main() -> None:
    bot = FAQChatbot()
    passed = 0
    for i, (query, expected) in enumerate(SAMPLES, 1):
        r = bot.respond(query)
        if expected is None:
            ok = r.intent in ("greeting", "fallback")
        else:
            ok = r.intent == "faq" and r.matched_question == expected
        passed += ok
        score = f"{r.score:.3f}" if r.score is not None else "  -  "
        print(f"{i:>2}. {'PASS' if ok else 'FAIL'}  [{r.intent:<8}] score={score}  Q: {query}")
        if r.intent == "faq":
            print(f"      matched: {r.matched_question}")
            print(f"      answer : {r.text[:90]}{'...' if len(r.text) > 90 else ''}")
        elif r.intent == "fallback":
            print("      suggestions: " + " | ".join(f"{q} ({s:.2f})" for q, s in r.suggestions))
        else:
            print(f"      reply  : {r.text}")
    print(f"\n{passed}/{len(SAMPLES)} behaved as expected (threshold={bot.threshold})")


if __name__ == "__main__":
    main()
