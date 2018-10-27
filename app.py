import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import cv2
import urllib
import numpy as np

import keras 
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(
    file_id='1zwPyJl-vFSxS7MCuUICmciCF235wt9Oy',
    dest_path='./my_model.hdf5')

def vgg16():
    # load pre-trained model graph, don't add final layer
    model = keras.applications.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    # add global pooling just like in InceptionV3
    new_output = keras.layers.Flatten()(model.output)
    # add new dense layer for our labels
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    new_output = keras.layers.Dense(512, activation='relu')(new_output)
    new_output = keras.layers.Dense(10, activation='softmax')(new_output)
    model = keras.engine.training.Model(model.inputs, new_output)
    return model
	
model = vgg16()
model.load_weights('./my_model.hdf5')

dict_num_cities = {0: 'chicago',
                   1: 'london',
                   2: 'losangeles',
                   3: 'melbourne',
                   4: 'miami',
                   5: 'newyork',
                   6: 'sanfrancisco',
                   7: 'singapore',
                   8: 'sydney',
                   9: 'toronto'}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Input(id='my-id', value='https://cdn.londonandpartners.com/visit/general-london/areas/river/76709-640x360-houses-of-parliament-and-london-eye-on-thames-from-above-640.jpg', type='text'),
    html.Img(id='my-div'),
	html.Div(id='my-div2')
])


@app.callback(
    Output(component_id='my-div', component_property='src'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_img(input_value):
    return input_value
	
@app.callback(
    Output(component_id='my-div2', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_img(input_value):
	req = urllib.request.urlopen(input_value)
	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
	img = cv2.imdecode(arr, -1)
	img = cv2.resize(img, (224,224))
	pred = model.predict(np.expand_dims(img, axis=0))
	return dict_num_cities[np.argmax(pred)]


if __name__ == '__main__':
    app.run_server()