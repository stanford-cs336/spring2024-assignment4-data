def gopher_quality_filter(text: str) -> bool:
    """Check if the text meets the quality criteria."""

    word_list = text.split()
    line_list = text.split("\n")

    # Check if the number of words is between 50 and 100,000
    if not (50 <= len(word_list) <= 100_000):
        return False

    # Check if the average word length is between 3 and 10 characters
    try:
        avg_word_len = sum(len(word) for word in word_list) / len(word_list)
        if not (3 <= avg_word_len <= 10):
            return False
    except ZeroDivisionError:
        return False

    # Check if more than 30% of the lines end with an ellipsis
    try:
        ellipsis_lines = [line for line in line_list if line.endswith("...")]
        if len(ellipsis_lines) / len(line_list) > 0.3:
            return False
    except ZeroDivisionError:
        return False

    # Check if more than 80% of the words contain at least one alphabetic character
    try:
        alphabetic_words = [word for word in word_list if any(char.isalpha() for char in word)]
        if len(alphabetic_words) / len(word_list) < 0.8:
            return False
    except ZeroDivisionError:
        return False

    return True
