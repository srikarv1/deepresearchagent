"""Conservative lexical screen, not a semantic classifier or search-keyword ban."""
import re

# Match commands directed at the agent. Nouns such as 'analysis of published
# experiments' and searches for these actions are deliberately not rejected.
_IMPERATIVE = re.compile(
    r'^(?:please\s+)?(?:'
    r'(?:run|conduct|deploy|instrument|simulate|calculate|estimate|analy[sz]e|'
    r'synthesi[sz]e|draft|negotiate|procure|purchase|build|implement|install|'
    r'create|produce|formulate|replicate|field[- ]validate)\b|'
    r'collect\s+(?:new\s+)?(?:measurements|data|samples)\b|'
    r'model\s+(?:the|a|an|our|this|coupled)\b|'
    r'derive\s+(?:an?\s+)?(?:operational|SOP|design|policy|estimate)\b)', re.I)
_SEARCH = re.compile(r'^(?:please\s+)?(?:find|search|retrieve|look\s+up|locate)\b|^run\s+(?:a\s+)?(?:web\s+|literature\s+)?search\b',re.I)


def check_retrieval(text):
    text=text.strip()
    if not _SEARCH.search(text) and _IMPERATIVE.search(text):
        raise ValueError('Input must be a retrieval query; obvious non-retrieval imperative requests an action instead of published evidence')


def check_followups(cards):
    for card in cards:
        for followup in card.followups:
            check_retrieval(followup)
            if not _SEARCH.search(followup) and re.search(r'^(?:please\s+)?(?:extract|compare|process|evaluate)\b',followup.strip(),re.I):
                raise ValueError('Follow-up must be a retrieval information need, not extracting/comparing/analyzing data ourselves')
