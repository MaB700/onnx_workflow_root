import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import xgboost as xgb
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType

boston = load_boston()

x, y = boston.data, boston.target
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.30, random_state=99)

model = xgb.XGBRegressor(objective='reg:squarederror',colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5,alpha = 10, n_estimators = 10)
print(model)

model.fit(xtrain, ytrain) 

test_in = np.full((1, 13), 0.2)
print(model.predict(test_in))

outtest = model.predict(xtest[0:100])

initial_types = [('float_input', FloatTensorType([None, xtrain.shape[1]]))]
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types)
onnxmltools.utils.save_model(onnx_model, './xgboost_boston.onnx')

np.savetxt("input_xgb.csv", xtest[0:100], delimiter=",")
np.savetxt("output_xgb.csv", outtest, delimiter=",")


