import NN_definer as nd
import keras
from keras.models import Sequential


class ensemble:


    model_list = []
    output_model = Sequential()#just a placeholder model
    neuralnet_d = nd.NN_definer()




    def define_ensemble(self ,model_name_list,output_model_name):

        for models in model_name_list:
            self.model_list.append(nd.load_model(models))

        self.output_model = nd.load_model(output_model_name)



    def predict(self,data):
        predictions = []
        for models in self.model_list:
         predictions.append(   self.neuralnet_d.make_single_prediction_with_model(models) )
        return predictions


    def predict_ensemble(self,data):

        initial_predictions = self.predict(data)

        for index, column  in enumerate(initial_predictions):
            data.append(data["column %"+1].values.entry(initial_predictions[index]))








