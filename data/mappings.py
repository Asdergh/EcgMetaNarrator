"""Mappings from archive-specific diagnosis tokens to canonical labels."""

CHALLENGE2020_CODE_MAP = {
    "426783006": "NORM",
    "164889003": "AF",
    "270492004": "IAVB",
    "164909002": "LBBB",
    "59118001": "RBBB",
    "284470004": "PAC",
    "427172004": "PVC",
    "429622005": "STD",
    "164931005": "STE",
}
CHALLENGE2020_PRIORITY = (
    "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE", "NORM",
)

NORMAL_DIAGNOSIS_TOKENS = {"NORM", "NORMAL", "426783006", "NORMAL_ECG"}
ABNORMAL_DIAGNOSIS_TOKENS = {
    "ABNORMAL", "MI", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC",
    "STD", "STE", "164865005", "164889003", "270492004", "164909002",
    "59118001", "284470004", "427172004", "429622005", "164931005",
}


def map_challenge2020_tokens(tokens: set[str]) -> str | None:
    mapped = {CHALLENGE2020_CODE_MAP.get(token, token) for token in tokens}
    return next((label for label in CHALLENGE2020_PRIORITY if label in mapped), None)


def map_binary_diagnosis(token: str) -> str | None:
    if token in NORMAL_DIAGNOSIS_TOKENS:
        return "NORMAL"
    if token in ABNORMAL_DIAGNOSIS_TOKENS:
        return "ABNORMAL"
    return None
