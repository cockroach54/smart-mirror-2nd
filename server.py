import os, base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, Response, render_template
import sys, json, random, time, argparse
import numpy as np
from time import sleep
import cv2
from preprocess import *
from detector2 import *
from cam_conv_model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=str, default='T', choices=['T','F'],
            help="Flask debug True or False. On demo set this False")
parser.add_argument('--port', type=int, default=5000,
            help="Flask server port")
parser.add_argument('--cam', type=str, default='F', choices=['T','F'],
            help="CAM model usage. True or False. default False")            

args = parser.parse_args()
PORT = args.port
DEBUG = True if args.d=='T' else False
USE_CAM = True if args.cam=='T' else False
print('[args]', args)

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
Load CAM model
"""     
if USE_CAM:       
    cam_model = CAM(n_classes=21)
    cam_model.to(cam_model.device)
    cam_model.eval()
    cam_model.load_state_dict(torch.load('cam-myconv-21.pth'))
else:
    cam_model = None


"""
Load Oracle model
"""
model = OracleModel()
model.to(model.device)
model.eval()
# make model dataset first 
image_ext_path = os.path.join('static', 'images_ext')
model.makeAllReference_online(image_ext_path)

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
    image_object = Image.open(image_file) # RGB
    if mirror=="true": frame = np.array(image_object)[:,::-1,:].copy() # horizental flip
    else: frame = np.array(image_object).copy() 
    if imageScale != 1.0: frame = cv2.resize(frame, dsize=(0, 0), fx=imageScale, fy=imageScale, interpolation=cv2.INTER_LINEAR) # scale image size
    # print('[frame]', frame.shape)

    im_tensor = model.roi_transform(frame).data.to(model.device).unsqueeze(0)
    featuremaps = model(im_tensor)
    # region proposal network extracts ROIs
    boxes = rpn2(frame, n_slice_x=rpnNumX, n_slice_y=rpnNumY, scale=rpnScale)
    # boxes, scores = rpn(frame, num_boxs=300, scale=0.5)

    # roi align
    _boxes_cuda = torch.from_numpy(boxes).float().cuda()
    rois = get_rois(im_tensor, featuremaps, _boxes_cuda)

    preds, preds_dist= model.inference_tensor3(rois, 'cos', knn=False)

    # objectness filterling
    filter_idx = (preds_dist[:,0]>model.threshold).type(torch.bool).cpu()
    if not any(filter_idx): 
        # continue # 필터 통과하는거 하나도 없을경우
        return [], frame
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
    bboxes_all = np.array(list(zip(boxes_cuda.cpu().detach().numpy(), preds.cpu().numpy()[:,0],
                                    preds_dist.cpu().numpy()[:,0])), dtype=np.object)
    bboxes_all_nms = []
    for cls in set(bboxes_all[:,1]):
        bboxes_all_nms.append(non_max_sup_one_class(bboxes_all[bboxes_all[:,1]==cls], threshold=0.1, descending=model.sort_order_descending))
    bboxes_all_nms = np.concatenate(bboxes_all_nms)
    
    if len(bboxes_all_nms) > 0:
        # render frame
        for idx, (box, pred, dist) in enumerate(bboxes_all_nms):        
            pred_label = model.reference_classes[pred]
            res_text = pred_label+"("+str(dist)+")"
            print(pred, ':', res_text, box)
    return bboxes_all_nms, frame

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def faceRecognition(req): 
    global face_cascade
    image_file = req['image']
    mirror = req['mirror']
    image_object = Image.open(image_file)
    if mirror=="true": frame = np.array(image_object)[:,::-1,::-1].copy() # RGB -> BGR for opencv, horizental flip
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

@app.route('/api/restart')
def api_restart():
    global image_ext_path
    model.makeAllReference_online(image_ext_path)

"""from video to image extraction"""
@app.route('/api/upload', methods=['POST'])
def api_upload():
    global image_ext_path, cam_model
    f = request.files['myVideo']
    filename = f.filename # 비디오 이름
    f.save("static/videos/"+filename)

    itemName = request.form.get('name')
    crop_scsle_ratio = request.form.get('cropScale')
    crop_scsle_ratio = list(map(lambda x: float(x), crop_scsle_ratio.split(',')))

    rootPath = "static"
    extractor = ImageExtractor(rootPath, filename)
    extractor.cam_model = cam_model
    # 비디오 ratio 결정 
    extractor.CONSTANT_RATIO = True
    # 비디오 리사이징 - crop 모드
    extractor.resizeVideo(mode="crop", crop_scsle_ratio=crop_scsle_ratio)
    # 비디오 전처리
    extractor.preprocessVideo(merge_ratio_limit=0.5, SHOW_IMAGE=False)
    # 통계량 추출 
    extractor.getStatistics(SHOW_PLOT=False)
    # # 결과 이미지 영역 크롭
    extractor.extractImages(interval=12, SHOW_IMAGE=False)
    # 비디오 추출시마다 레퍼런스 디비 재생성
    # model.makeAllReference_online(image_ext_path)
    model.addNewLabel_online(image_ext_path, itemName)
    return json.dumps({'success': True, 'filename': filename})

"""image detection api"""
confuse_cnt=0 # 헷갈린 개수
@app.route('/api/detectweb', methods=['POST'])
def api_detectweb():
    req = {}
    req['image'] = request.files['image']
    req['confuseFlag'] = True if request.form.get('confuseFlag')=='true' else False
    req['threshold'] = float(request.form.get('threshold_c')) if req['confuseFlag'] else float(request.form.get('threshold')) # 일단 confuse 기준으로 전부 필터링
    req['interval_c'] = int(request.form.get('interval_c'))
    req['mirror'] = request.form.get('mirror')
    req['imageScale'] = float(request.form.get('imageScale'))
    req['rpnNumX'] = int(request.form.get('rpnNumX'))
    req['rpnNumY'] = int(request.form.get('rpnNumY'))
    req['rpnScale'] = [float(request.form.get('rpnScaleX')), float(request.form.get('rpnScaleY'))]
    # bboxes_all_nms = detect_boxes(req)

    # return json.dumps({
    #     'success': True,
    #     'labels': model.reference_classes,
    #     'bboxes': bboxes_all_nms
    # }, cls=MyEncoder)    

    bboxes_all_nms, frame = detect_boxes(req)
    
    if(req['confuseFlag']):
        frames = []
        global confuse_cnt
        if(len(bboxes_all_nms)>0):
            mask = bboxes_all_nms[:,2]> float(request.form.get('threshold'))
            mask_invert = [not m for m in mask]
            bboxes_all_nms_confused = bboxes_all_nms[mask_invert]
            bboxes_all_nms = bboxes_all_nms[mask]
            if np.all(mask_invert): # 헷갈리는 것만 있으면
                print(mask_invert)
                confuse_cnt += 1
                if confuse_cnt>req['interval_c']: # n번에 한번만 보냄
                    confuse_cnt = 0
                    # 헷갈리는 영역 이미지 crop해서 base64로 인코딩
                    for box in bboxes_all_nms_confused:
                        x,y,w,h = box[0].astype(int)
                        idx = box[1]
                        f = frame[y:y+h, x:x+w]
                        f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_CUBIC)
                        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) # opencv image need to convert BGR -> RGB
                        _cnt = cv2.imencode('.jpg',f)[1].flatten()
                        b64 = base64.encodebytes(_cnt).decode() # rgb
                        frames.append([idx, b64])
                    return json.dumps({
                        'success': True,
                        'labels': model.reference_classes,
                        'bboxes': bboxes_all_nms,
                        'confused': True,
                        'frames': frames
                    }, cls=MyEncoder)    
    
    return json.dumps({
        'success': True,
        'labels': model.reference_classes,
        'bboxes': bboxes_all_nms,
        'confused': False
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
    bboxes_all_nms, frame = detect_boxes(req)
    # for saving input image
    im = Image.open(req['image'])
    if req['mirror']=="true": im = np.array(im)[:,::-1,:].copy() # horizental flip
    else: im = np.array(im).copy() 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
    cv2.imwrite('predict.jpg', im)
    # for plot image
    plotName = 'plot-'+str(int(time.time()))+'.jpg'
    plotPath = os.path.join('static','images','plots',plotName)
    plotPath_for_client = os.path.join('images','plots',plotName)
    model.save_plot(plotPath)

    print('[Model classes]:', model.reference_classes)

    return json.dumps({
        'success': True,
        'labels': model.reference_classes,
        'bboxes': bboxes_all_nms,
        'plotPath': plotPath_for_client
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

"""save confused image api"""
@app.route('/api/confused', methods=['POST'])
def api_confused():
    global image_ext_path, USE_CAM
    b64 = request.form.get('image_b64')
    idx = int(request.form.get('idx'))
    label = model.reference_classes[idx]
    # save image to extract folder
    filename = label+'-'+str(int(time.time()))+'.jpg'
    imagePath = os.path.join(image_ext_path, label, filename)
    imgbyte = base64.b64decode(b64)
    # imgbyte to np array
    nparr = np.frombuffer(imgbyte, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # bgr
    img_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) # rgb
    # mask cam image
    if USE_CAM:
        images_normal = cam_model.trans_normal(img_np).unsqueeze(0)
        images_normal = images_normal.to(cam_model.device)
        cams_scaled, masks_np = cam_model.getCAM(images_normal)
        img_cv = img_cv*masks_np[0]
        img_np = img_np*masks_np[0]

    cv2.imwrite(imagePath, img_cv)
    print('[Confused]: Save image - ', imagePath)    
    # remake reference data (RGB)
    model.addNewData_online(img_np, label)
    # model.makeAllReference_online(image_ext_path)
    return json.dumps({
        'success': True,
        'imagePath': imagePath
    }, cls=MyEncoder)

"""trim and save image api"""
@app.route('/api/savetrim', methods=['POST'])
def api_savetrim():
    global image_ext_path, USE_CAM
    req = {}
    req['image'] = request.files['image']
    req['mirror'] = request.form.get('mirror')
    itemName = request.form.get('name')
    coords_r = list(map(lambda x: float(x), request.form.get('coords').split(','))) # rx1,ry1,rx2,ry2

    im = Image.open(req['image'])
    if req['mirror']=="true": img_np = np.array(im)[:,::-1,:].copy() # horizental flip, rgb
    else: img_np = np.array(im).copy() # rgb
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # bgr
    coords = [coords_r[0]*img_cv.shape[1], coords_r[1]*img_cv.shape[0], coords_r[2]*img_cv.shape[1], coords_r[3]*img_cv.shape[0]]
    coords = np.array(coords, dtype=np.int)
    print('[Trim coords]', coords, img_cv.shape)
    img_cv = img_cv[coords[1]:coords[3], coords[0]:coords[2]] # trim

    # mask cam image
    if USE_CAM:
        images_normal = cam_model.trans_normal(img_np).unsqueeze(0)
        images_normal = images_normal.to(cam_model.device)
        cams_scaled, masks_np = cam_model.getCAM(images_normal)
        img_cv = img_cv*masks_np[0]
        img_np = img_np*masks_np[0]

    # save image
    cv2.imwrite('savetrim.jpg', img_cv) # for debug
    dirPath = os.path.join(image_ext_path, itemName)
    isExist = os.path.isdir(dirPath)

    # new label
    if(not isExist):
        print('[Savetrim]: make new dir', itemName)
        os.mkdir(dirPath)

    # thumbnail용 이미지 먼저 확인
    fileName = itemName+'_1.jpg'
    imagePath = os.path.join(dirPath, fileName)
    if(os.path.exists(imagePath)):
        fileName = itemName+'-'+str(int(time.time()))+'.jpg'
        imagePath = os.path.join(dirPath, fileName)        
    cv2.imwrite(imagePath, img_cv)
    print('[Savetrim]: Save image - ', imagePath)    

    # already exist label
    if(isExist): model.addNewData_online(img_np, itemName)
    else: model.addNewLabel_online(image_ext_path, itemName)

    return json.dumps({
        'success': True,
        'imagePath': imagePath,
        'newLabel': not isExist
    }, cls=MyEncoder)


if __name__ == '__main__':
	# without SSL
    # app.run(debug=DEBUG, host='0.0.0.0', port=PORT, threaded=True)

	# with SSL - debug, threded를 false로 하면 시연시 속도가 빨라짐
    print('[DEBUG]:', DEBUG)
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT, threaded=False, ssl_context=('ssl/cert.pem', 'ssl/key.pem'))