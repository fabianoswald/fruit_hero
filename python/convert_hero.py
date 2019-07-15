# USAGE
# python coremlconverter.py --model fruit_hero.model --labelbin lb.pickle

# import necessary packages
from keras.models import load_model
import coremltools
import argparse
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())

# load the class labels
print("[INFO] loading class labels from label binarizer")
lb = pickle.loads(open(args["labelbin"], "rb").read())
class_labels = lb.classes_.tolist()
print("[INFO] class labels: {}".format(class_labels))

# load the trained convolutional neural network
print("[INFO] loading model...")
model = load_model(args["model"])

# convert the model to coreml format
print("[INFO] converting model")
coreml_model = coremltools.converters.keras.convert(model,
	input_name_shape_dict={"image" : [None,96,96,3]},                        
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	class_labels=class_labels,
	is_bgr=True)

# save the model to disk
output = args["model"].rsplit(".", 1)[0] + ".mlmodel"
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)

#coremltools.utils.save_spec(coreml_model, 'fruithero.mlmodel')
