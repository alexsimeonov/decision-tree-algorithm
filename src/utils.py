def filter_dictionary(dictionary, condition):
    filtered_dictionary = {}

    for (key, value) in dictionary.items():
        if condition(value):
            # currently the keys are not in order and taken from the initial dictionary
            # do we have to keep the data in dictionary again or we can simply parse it to list instead?
            filtered_dictionary[key] = value

    return filtered_dictionary
