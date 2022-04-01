
def create_model(opt):
    from .sscnn_model import CNNModel
    model = CNNModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
