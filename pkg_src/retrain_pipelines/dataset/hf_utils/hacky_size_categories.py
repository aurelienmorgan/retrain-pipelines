
import re
import requests


def _get_size_categories():
    """
    The acceptable values can't be retrieved
    from the huggingface_hub API.
    So, we dirty-hack retrieve it
    from source code docstring..
    """

    url = "https://raw.githubusercontent.com/huggingface/huggingface_hub/main/src/huggingface_hub/repocard_data.py"
    response = requests.get(url)
    content = response.text
    
    pattern = r"size_categories.*?Options are: (.*?)\."
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        categories_str = match.group(1)
        categories = re.findall(r"'([^']*)'", categories_str)
        return categories
    else:
        return []


def _parse_number(s: str):
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    if s == 'n':
        return None
    for suffix, multiplier in multipliers.items():
        if suffix in s:
            return float(s.replace(suffix, '')) * multiplier
    return float(s)


def _convert_to_bounds(
    categories_str_list: list
) -> list:
    """
    from list of human-friendly strings of intervals
    (e.g. "100K<n<1M").
    """

    bounds = []
    for category in categories_str_list:
        if category == "other":
            bounds.append(("other", "other"))
        elif "<" in category:
            parts = category.split("<")
            if parts[0] == "n":
                lower, upper = None, _parse_number(parts[1])
            else:
                lower, upper = \
                    _parse_number(parts[0]), \
                    _parse_number(parts[2])
            bounds.append((lower, upper))
        elif ">" in category:
            lower = _parse_number(category.split(">")[1])
            bounds.append((lower, None))
    return bounds


def get_size_category(
    dataset_records_count: int
):
    """
    """

    try:
        size_categories = _get_size_categories()
        bounds = _convert_to_bounds(size_categories)

        for category, (lower, upper) \
        in zip(size_categories, bounds):
            if category == "other":
                continue
            if (
                (
                    lower is None or
                    dataset_records_count >= lower
                ) and (
                    upper is None or
                    dataset_records_count < upper
                )
            ):
                return category
        return "other"
    except Exception:
        return "unknown"

