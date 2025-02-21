from src import DummyModel
if __name__ == "__main__":
    model = DummyModel()
    print('The model:')
    print(model)

    print("\n\nJust one layer:")
    print(model.fc)

    print("\n\nModel params:")
    for param in model.parameters():
        print(param)

    print("\n\nLayer params:")
    for param in model.fc.parameters():
        print(param)
