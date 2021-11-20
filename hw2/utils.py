import pickle


def save_pickle(data, path: str):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(path: str):
    with open(path, "rb") as file:
        return pickle.load(file)
