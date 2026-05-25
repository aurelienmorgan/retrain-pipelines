import re

import requests

Bounds = tuple[float | None, float | None]


def _get_size_categories() -> list[str]:
    """Retrieve acceptable size categories for README metadata yaml header.

    The acceptable values can't be retrieved
    from the huggingface_hub API.
    So, we dirty-hack retrieve it
    from source code docstring.
    """
    url = (
        "https://raw.githubusercontent.com/huggingface/"
        "huggingface_hub/main/src/huggingface_hub/repocard_data.py"
    )

    response = requests.get(url)
    response.raise_for_status()

    content = response.text

    pattern = r"size_categories.*?Options are: (.*?)\."
    match = re.search(pattern, content, re.DOTALL)

    if match:
        categories_str = match.group(1)
        categories = re.findall(r"'([^']*)'", categories_str)
        return categories

    return []


def _parse_number(s: str) -> float | None:
    multipliers = {
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
        "T": 1e12,
    }

    if s == "n":
        return None

    for suffix, multiplier in multipliers.items():
        if suffix in s:
            return float(s.replace(suffix, "")) * multiplier

    return float(s)


def _convert_to_bounds(categories_str_list: list[str]) -> list[Bounds]:
    """Convert human-friendly interval strings to numeric bounds.

    Examples
    --------
    >>> "100K<n<1M" -> (100000.0, 1000000.0)
    """
    bounds: list[Bounds] = []

    for category in categories_str_list:
        if category == "other":
            bounds.append((None, None))

        elif "<" in category:
            parts = category.split("<")

            if parts[0] == "n":
                lower = None
                upper = _parse_number(parts[1])
            else:
                lower = _parse_number(parts[0])
                upper = _parse_number(parts[2])

            bounds.append((lower, upper))

        elif ">" in category:
            lower = _parse_number(category.split(">")[1])
            bounds.append((lower, None))

    return bounds


def get_size_category(dataset_records_count: int) -> str:
    try:
        size_categories = _get_size_categories()
        bounds = _convert_to_bounds(size_categories)

        for category, (lower, upper) in zip(
            size_categories,
            bounds,
            strict=False,
        ):
            if category == "other":
                continue

            if (lower is None or dataset_records_count >= lower) and (
                upper is None or dataset_records_count < upper
            ):
                return category

        return "other"

    except Exception:
        return "unknown"
