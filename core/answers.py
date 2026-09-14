"""Answer extraction shared by the structured-input runner."""
import re


def final_text(text):
    text = "" if text is None else str(text).strip()
    if "<think" in text.lower() and "</think>" not in text.lower():
        return ""
    text = re.split(r"</think\s*>", text, flags=re.I)[-1].strip()
    answers = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if answers:
        return answers[-1].strip()
    answers = re.findall(r"Final\s+Answer\s*[:：]\s*([^\r\n]+)", text, re.I)
    if answers:
        return answers[-1].strip().strip("*` ")
    return text


def choice_letter(text, choices):
    text = final_text(text)
    if not text:
        return None
    match = re.fullmatch(r"\s*\(?([A-Z])\)?[.)]?\s*", text, re.I)
    if match and match[1].upper() in choices:
        return match[1].upper()
    hits = [key for key, value in choices.items()
            if str(value).strip().casefold() == text.casefold()]
    if len(hits) == 1:
        return hits[0]
    # Do not mine incidental option letters from a chain of thought.
    match = re.fullmatch(r"(?:Answer|Option)\s*[:=]?\s*\(?([A-Z])\)?[.!]?", text, re.I)
    return match[1].upper() if match and match[1].upper() in choices else None
