

class FederatedServer():
    def __init__(self, model_fn):
        self._model = model_fn()

    @property
    def server_weights(self):
        return self._model.get_weights()

    def server_update(self, mean_client_weights):
        # Assign the mean client weights to the server model
        self._model.set_weights(mean_client_weights)

        return self._model.get_weights()

