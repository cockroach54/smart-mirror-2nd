import os, base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, Response, render_template
import sys, json, random, time, argparse
import numpy as np
import cv2
from preprocess import *
from detector import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='T', choices=['T','F'],
            help="Flask debug True or False. On demo set this False")
parser.add_argument('--port', type=int, default=5000,
            help="Flask server port")            

args = parser.parse_args()
PORT = args.port
DEBUG = True if args.d=='T' else False

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

"""
Load Oracle model
"""
model = OracleModel()
model.to(model.device)
model.eval()
# make model dataset first 
model.makeAllReference_online('static/images_ext')

# """
# Load UBBR model
# now not used for speed
# """
# LOAD_PATH = '../../torch_models/ubbr-iou-max-10.pt'
# UBBR_model = torch.load(LOAD_PATH)


def detect_boxes(req):
    image_file = req['image']
    threshold = req['threshold']
    mirror = req['mirror']
    imageScale = req['imageScale']
    rpnNumX = req['rpnNumX']
    rpnNumY = req['rpnNumY']
    rpnScale = req['rpnScale']

    model.threshold = threshold

    # image array preprocess
    image_object = Image.open(image_file)
    if mirror=="true": frame = np.array(image_object)[:,::-1,::-1].copy() # RGB -> BGR for opencv, vertical flip
    else: frame = np.array(image_object)[:,:,::-1].copy() # RGB -> BGR for opencv
    if imageScale != 1.0: frame = cv2.resize(frame, dsize=(0, 0), fx=imageScale, fy=imageScale, interpolation=cv2.INTER_LINEAR) # scale image size
    print('[frame]', frame.shape)

    im_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # opencv image need to convert BGR -> RGB
    im_tensor = model.roi_transform(im_tensor).data.to(model.device).unsqueeze(0)
    featuremaps = model(im_tensor)
    # region proposal network extracts ROIs
    # boxes = rpn2(frame, n_slice_x=rpnNumX, n_slice_y=rpnNumY, scale=rpnScale)
    boxes = rpn(frame, num_boxs=200, scale=.5)

    # roi align
    _boxes_cuda = torch.from_numpy(boxes).float().cuda()
    rois = get_rois(im_tensor, featuremaps, _boxes_cuda)

    preds, preds_dist= model.inference_tensor3(rois, 'cos', knn=True)

    # objectness filterling
    filter_idx = (preds_dist[:,0]>model.threshold).type(torch.bool).cpu()
    if not any(filter_idx): 
        # continue # 필터 통과하는거 하나도 없을경우
        return []
    _boxes_cuda = _boxes_cuda[filter_idx]
    preds = preds[filter_idx]
    preds_dist = preds_dist[filter_idx]
    rois = rois[filter_idx]

    boxes_cuda = _boxes_cuda.clone().detach()

    # # UBBR adjusted bboxes
    # offsets = UBBR_model.fc(rois) 
    #             # offsets = UBBR_model(im_tensor, offsets)        
    # # reg 적용한 random boxes
    # boxes_cuda = regression_transform(boxes_cuda, offsets)

    # non-maximum-suppression
    bboxes_all = np.array(list(zip(boxes_cuda.cpu().detach().numpy(), preds.cpu().numpy()[:,0], preds_dist.cpu().numpy()[:,0])), dtype=np.object)
    bboxes_all_nms = []
    for cls in set(bboxes_all[:,1]):
        bboxes_all_nms.append(non_max_sup_one_class(bboxes_all[bboxes_all[:,1]==cls], threshold=0.1, descending=model.sort_order_descending))
    bboxes_all_nms = np.concatenate(bboxes_all_nms)
    
    if len(bboxes_all_nms) > 0:
        # render frame
        for idx, (box, pred, dist) in enumerate(bboxes_all_nms):        
            pred_label = model.reference_classes[pred]
            res_text = pred_label+"("+str(dist)+")"
            x, y, w, h = box
            print(idx, ':', res_text, box)
    return bboxes_all_nms

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def faceRecognition(req): 
    global face_cascade
    image_file = req['image']
    mirror = req['mirror']
    image_object = Image.open(image_file)
    if mirror=="true": frame = np.array(image_object)[:,::-1,::-1].copy() # RGB -> BGR for opencv, vertical flip
    else: frame = np.array(image_object)[:,:,::-1].copy() # RGB -> BGR for opencv
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2.08, minNeighbors=3)
    if(len(faces)>0):
        print('[face]', faces)
    return faces

# ------------------------flask app------------------------------

app = Flask(__name__, static_url_path='',
            static_folder='./static',
            template_folder='./templates')

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response

"""home page"""
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

"""from video to image extraction"""
@app.route('/api/upload', methods=['POST'])
def api_upload():
    f = request.files['myVideo']
    filename = f.filename # 비디오 이름
    f.save("static/videos/"+filename)

    itemName = request.form.get('name')
    rootPath = "static"
    extractor = ImageExtractor(rootPath, filename)
    # 비디오 ratio 결정 
    extractor.CONSTANT_RATIO = True
    # 비디오 리사이징
    extractor.resizeVideo()
    # 비디오 전처리
    extractor.preprocessVideo(SHOW_IMAGE = False)
    # 통계량 추출 
    extractor.getStatistics(SHOW_PLOT=False)
    # # 결과 이미지 영역 크롭
    extractor.extractImages(SHOW_IMAGE=False)
    extractor.cap.release()
    # 비디오 추출시마다 레퍼런스 디비 재생성
    model.makeAllReference_online('static/images_ext')
    return json.dumps({'success': True, 'filename': filename})

"""image detection api"""
@app.route('/api/detectweb', methods=['POST'])
def api_detectweb():
    req = {}
    req['image'] = request.files['image']
    req['threshold'] = float(request.form.get('threshold'))
    req['mirror'] = request.form.get('mirror')
    req['imageScale'] = float(request.form.get('imageScale'))
    req['rpnNumX'] = int(request.form.get('rpnNumX'))
    req['rpnNumY'] = int(request.form.get('rpnNumY'))
    req['rpnScale'] = [float(request.form.get('rpnScaleX')), float(request.form.get('rpnScaleY'))]
    bboxes_all_nms = detect_boxes(req)

    return json.dumps({
        'success': True,
        'labels': model.reference_classes,
        'bboxes': bboxes_all_nms
    }, cls=MyEncoder)    
 
"""image detection api(make plot also)"""
@app.route('/api/infer', methods=['POST'])
def api_infer():
    req = {}
    req['image'] = request.files['image']
    req['threshold'] = float(request.form.get('threshold'))
    req['mirror'] = request.form.get('mirror')
    req['imageScale'] = float(request.form.get('imageScale'))
    req['rpnNumX'] = int(request.form.get('rpnNumX'))
    req['rpnNumY'] = int(request.form.get('rpnNumY'))
    req['rpnScale'] = [float(request.form.get('rpnScaleX')), float(request.form.get('rpnScaleY'))]
    im = Image.open(req['image'])
    im.save('predict.jpg')
    bboxes_all_nms = detect_boxes(req)
    model.save_plot()

    return json.dumps({
        'success': True,
        'labels': model.reference_classes,
        'bboxes': bboxes_all_nms
    }, cls=MyEncoder)    

""""for face recognition"""
@app.route('/api/faceRecog', methods=['POST'])
def api_faceRecog():
    req={}
    req['image'] = request.files['image']
    req['mirror'] = request.form.get('mirror')
    faces = faceRecognition(req)
    return json.dumps({
        'success': True,
        'face': len(faces),
        'bboxes': faces
    }, cls=MyEncoder)    


if __name__ == '__main__':
	# without SSL
    # app.run(debug=DEBUG, host='0.0.0.0', port=PORT, threaded=True)

	# with SSL - debug, threded를 false로 하면 시연시 속도가 빨라짐
    print('[DEBUG]:', DEBUG)
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT, threaded=False, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))