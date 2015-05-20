import beacon8 as bb8


def net():
    model = bb8.Sequential()
    model.add(bb8.Linear(28*28, 100))
    model.add(bb8.ReLU())

    model.add(bb8.Linear(100, 100))
    model.add(bb8.ReLU())

    model.add(bb8.Linear(100, 100))
    model.add(bb8.ReLU())

    model.add(bb8.Linear(100, 10))
    model.add(bb8.SoftMax())
    return model

