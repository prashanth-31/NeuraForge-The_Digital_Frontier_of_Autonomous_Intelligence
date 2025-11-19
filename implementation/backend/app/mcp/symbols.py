from __future__ import annotations

from typing import Final

KEYWORD_SYMBOL_OVERRIDES: Final[dict[str, list[str]]] = {
    "005930": ["005930.KS"],
    "aapl": ["AAPL"],
    "adbe": ["ADBE"],
    "adobe": ["ADBE"],
    "advanced micro devices": ["AMD"],
    "alibaba": ["BABA", "9988.HK"],
    "alibaba group": ["BABA", "9988.HK"],
    "alphabet": ["GOOGL", "GOOG"],
    "alphabet inc": ["GOOGL", "GOOG"],
    "amazon": ["AMZN"],
    "amazon.com": ["AMZN"],
    "amd": ["AMD"],
    "amzn": ["AMZN"],
    "apple": ["AAPL"],
    "arm": ["ARM"],
    "arm holdings": ["ARM"],
    "arm holdings plc": ["ARM"],
    "asml": ["ASML"],
    "asml holding": ["ASML"],
    "avgo": ["AVGO"],
    "baba": ["BABA", "9988.HK"],
    "baidu": ["BIDU"],
    "baidu inc": ["BIDU"],
    "bidu": ["BIDU"],
    "bp": ["BP", "BP.L"],
    "bp plc": ["BP", "BP.L"],
    "broadcom": ["AVGO"],
    "broadcom inc": ["AVGO"],
    "crm": ["CRM"],
    "facebook": ["META"],
    "google": ["GOOGL", "GOOG"],
    "hdfc": ["HDFCBANK.NS"],
    "ibm": ["IBM"],
    "icici": ["ICICIBANK.NS"],
    "infosys": ["INFY"],
    "infy": ["INFY"],
    "intel": ["INTC"],
    "international business machines": ["IBM"],
    "intc": ["INTC"],
    "lyft": ["LYFT"],
    "meta": ["META"],
    "microsoft": ["MSFT"],
    "msft": ["MSFT"],
    "netflix": ["NFLX"],
    "nvidia": ["NVDA"],
    "nvidia corp": ["NVDA"],
    "nvda": ["NVDA"],
    "oracle": ["ORCL"],
    "orcl": ["ORCL"],
    "qualcomm": ["QCOM"],
    "qualcomm inc": ["QCOM"],
    "qcom": ["QCOM"],
    "reliance": ["RELIANCE.NS"],
    "salesforce": ["CRM"],
    "salesforce inc": ["CRM"],
    "samsung": ["005930.KS", "SSNLF"],
    "samsung electronics": ["005930.KS"],
    "shell": ["SHEL"],
    "shop": ["SHOP"],
    "shopify": ["SHOP"],
    "shopify inc": ["SHOP"],
    "siemens": ["SIE.DE"],
    "snow": ["SNOW"],
    "snowflake": ["SNOW"],
    "snowflake inc": ["SNOW"],
    "softbank": ["9984.T"],
    "sony": ["6758.T", "SONY"],
    "ssnlf": ["SSNLF"],
    "taiwan semiconductor": ["TSM"],
    "taiwan semiconductor manufacturing": ["TSM"],
    "tencent": ["0700.HK", "TCEHY"],
    "tencent holdings": ["0700.HK", "TCEHY"],
    "tcehy": ["TCEHY"],
    "tesla": ["TSLA"],
    "tesla motors": ["TSLA"],
    "tm": ["TM"],
    "toyota": ["TM", "7203.T"],
    "toyota motor": ["TM", "7203.T"],
    "totalenergies": ["TTE"],
    "tsla": ["TSLA"],
    "tsm": ["TSM"],
    "tsmc": ["TSM"],
    "uber": ["UBER"],
    "uber technologies": ["UBER"],
    "volkswagen": ["VOW3.DE"],
}

_SYMBOL_TOKEN_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
    ("-", " "),
    ("/", " "),
    (",", " "),
    (";", " "),
    ("|", " "),
    ("\n", " "),
    ("\t", " "),
)

_ALLOWED_SYMBOL_CHARS: Final[set[str]] = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=/")


def keyword_symbol_lookup(normalized_query: str) -> list[str]:
    if not normalized_query:
        return []
    cleaned = normalized_query
    for source, target in _SYMBOL_TOKEN_REPLACEMENTS:
        cleaned = cleaned.replace(source, target)
    matches: list[str] = []
    tokens = cleaned.split()
    for token in tokens:
        symbols = KEYWORD_SYMBOL_OVERRIDES.get(token)
        if not symbols:
            continue
        for symbol in symbols:
            if symbol not in matches:
                matches.append(symbol)
    if matches:
        return matches
    for keyword, symbols in KEYWORD_SYMBOL_OVERRIDES.items():
        if keyword in cleaned:
            for symbol in symbols:
                if symbol not in matches:
                    matches.append(symbol)
    return matches


def looks_like_symbol(value: str) -> bool:
    candidate = value.strip().upper()
    if not candidate or len(candidate) > 12:
        return False
    if any(ch.isspace() for ch in candidate):
        return False
    return all(ch in _ALLOWED_SYMBOL_CHARS for ch in candidate)


def extract_symbols_from_text(value: str | None) -> list[str]:
    if not value:
        return []
    stripped = value.strip()
    if not stripped:
        return []
    if looks_like_symbol(stripped):
        return [stripped.upper()]
    matches = keyword_symbol_lookup(stripped.lower())
    return [symbol.upper() for symbol in matches]


__all__ = [
    "KEYWORD_SYMBOL_OVERRIDES",
    "keyword_symbol_lookup",
    "looks_like_symbol",
    "extract_symbols_from_text",
]
