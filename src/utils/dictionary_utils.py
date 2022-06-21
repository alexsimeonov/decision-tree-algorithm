def filter_dictionary(dictionary, condition):
    filtered_dictionary = {}

    for (key, value) in dictionary.items():
        if condition(value):
            filtered_dictionary[key] = value

    return filtered_dictionary
