import NeuralNetDefiner as Nnd
from keras.models import Sequential


class Ensemble:
    """ This class was to be for holding an ensemble model,this is incomplete for now so don't use it."""

    model_list = []
    output_model = Sequential()  # just a placeholder model
    neuralnet_d = Nnd.NeuralNetDefiner()

    def define_ensemble(self, model_name_list, output_model_name):
        """ This takes in a list of strings of the saved models, and the name of the output model"""
        for models in model_name_list:
            self.model_list.append(Nnd.load_model(models))

        self.output_model = Nnd.load_model(output_model_name)

    def predict(self, data):
        predictions = []
        for models in self.model_list:
            predictions.append(self.neuralnet_d.make_single_prediction_with_model(models))
        return predictions

    def predict_ensemble(self, data):

        initial_predictions = self.predict(data)

        for index, column in enumerate(initial_predictions):
            data.append(data["column %" + 1].values.entry(initial_predictions[index]))
