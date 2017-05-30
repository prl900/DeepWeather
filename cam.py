import numpy as np
from keras.models import load_model
import keras.backend as K
from PIL import Image
import matplotlib.cm as cm

airports = ['EIDW', 'EGLL', 'LFPG', 'LFBO', 'EGPH', 'EHAM', 'EBBR', 'LEMD', 'LEBL', 'LPPT', 'LIRF',
            'LIMC', 'LSZH', 'EDDM', 'EDFH', 'EDDT', 'EKCH', 'ENGM', 'ESSA', 'EFHK', 'LOWW']

def get_rains(codes):
    arr = np.load("data/rain.npy")
    idxs = [airports.index(code) for code in codes]
    return arr[:, idxs].astype(np.float32)

def get_era_full(param, level):
    #arr = np.load("data/{}{}_uint8.npy".format(param, level))[[0], :, :]
    arr = np.load("data/{}{}_uint8.npy".format(param, level))
    #arr = np.load("data/{}{}_low_uint8.npy".format(param, level))[[0], :, :]
    return arr.astype(np.float32) / 256.

def get_cam(airport):
    model = load_model('model_{}.h5'.format(airport))

    dry_weights = model.layers[-1].get_weights()[0][:, 0].reshape((20,30,256))
    rain_weights = model.layers[-1].get_weights()[0][:, 1].reshape((20,30,256))
    get_output = K.function([model.layers[0].input], [model.layers[-3].output, model.layers[-1].output])

    cam = np.zeros((20,30), dtype=np.float32)
    for depth in range(rain_weights.shape[2]):
        for i in range(30):
            for j in range(20):
                cam[j, i] += rain_weights[j, i, depth]

    cam /= np.max(cam)
    im = Image.fromarray(np.uint8(cm.jet(cam)*255))
    im = im.resize((120,80), Image.ANTIALIAS)
    #heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    im.save("{}_rain.png".format(airport))

if __name__ == '__main__':
    airports = ['EIDW', 'EGLL', 'LFPG', 'LFBO', 'LOWW', 'EFHK', 'EGPH', 'EHAM', 'EBBR', 'ENGM', 'LEMD', 'LEBL', 'LPPT', 'LIRF']
    for airport in airports:
        get_cam(airport)
