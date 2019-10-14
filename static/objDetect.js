
let threshold = document.querySelector("#threshold");
let imageScale = document.querySelector("#imageScale");
let rpnNumX = document.querySelector('#rpnNumX');
let rpnNumY = document.querySelector('#rpnNumY');
let rpnScaleX = document.querySelector('#rpnScaleX');
let rpnScaleY = document.querySelector('#rpnScaleY');

let radioEl = document.getElementsByName('radio-for-box');
let onlyOneEl = document.querySelector('#onlyOne');

// ********************* Get camera video **********************
const constraints = {
    audio: false,
    // video: {
    //     width: {min: 640, ideal: 1280, max: 1920},
    //     height: {min: 480, ideal: 720, max: 1080}
    // }
    video: {
        // facingMode: "environment", // mobile에서 적용 "environment":후면카메라, "user":전면카메라
        facingMode: "user", // mobile에서 적용 "environment":후면카메라, "user":전면카메라
        width: { min: 100, ideal: 420, max: 640 },
        // height: {min: 200, ideal: 360, max: 480}
    }
};

navigator.mediaDevices.getUserMedia(constraints)
    .then(stream => {
        window.stream = stream;
        document.getElementById("myVideo").srcObject = stream;
        console.log("Got local user video");

    })
    .catch(err => {
        console.log('navigator.getUserMedia error: ', err)
    });
// ***************************************************************

// mobile setup
let mirror = true;
let reg_phone = /Android|webOS|iPhone|iPad|iPod|BlackBerry|BB|PlayBook|IEMobile|Windows Phone|Kindle|Silk|Opera Mini/i;
if (reg_phone.test(navigator.userAgent)) {
    rpnNumX.value = 1;
    rpnNumY.value = 1;
    rpnScaleX.value = 0.6;
    rpnScaleY.value = 0.45;
    threshold.value = 0.88;
    // Take the user to a different screen here.
    // 모바일 후면 카메라 미러효과 없애기
    if(constraints.video.facingMode==="environment"){
        document.querySelector('video').style.transform = 'scale(1,1)';
        mirror = false
    }
}

//Parameters
const s = document.getElementById('objDetect');
const sourceVideo = s.getAttribute("data-source");  //the source video to use
const uploadWidth = s.getAttribute("data-uploadWidth") || 640; //the width of the upload file
//Video element selector
v = document.getElementById(sourceVideo);

//for starting events
let isPlaying = false,
    gotMetadata = false;

//Canvas setup

//create a canvas to grab an image for upload
let imageCanvas = document.createElement('canvas');
let imageCtx = imageCanvas.getContext("2d");

//create a canvas for drawing object boundaries
let drawCanvas = document.createElement('canvas');
let videoWrapper = document.querySelector("#videoWrapper");
videoWrapper.appendChild(drawCanvas);
let drawCtx = drawCanvas.getContext("2d");
drawCtx.fillStyle = "#ffffff";
drawCtx.strokeStyle = "#ffff00";

// -------------------------  detection apic -----------------------------
function setForDetect(url){
    return new Promise((resolve, reject) => {
        imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
        imageCanvas.toBlob((file) => {
            let formdata = new FormData();
            formdata.append("image", file);
            formdata.append("threshold", threshold.value);
            formdata.append("imageScale", imageScale.value);
            formdata.append("rpnNumX", rpnNumX.value);
            formdata.append("rpnNumY", rpnNumY.value);
            formdata.append("rpnScaleX", rpnScaleX.value);
            formdata.append("rpnScaleY", rpnScaleY.value);
            formdata.append("mirror", mirror);
            let myInit = {
                method: 'POST',
                body: formdata
            }
    
            fetch(url, myInit).then((res) => {
                if (res.status === 200 || res.status === 201) { // 성공을 알리는 HTTP 상태 코드면
                    resolve(res.json())
                } else { // 실패를 알리는 HTTP 상태 코드면
                    console.error(res.statusText);
                    reject(res.statusText);
                }
            })
        }, 'image/jpeg');
    });
}

// inference button event
let inferBtn = document.querySelector('#infer');
inferBtn.addEventListener("click", ()=>{
    document.querySelector('#pred').innerText = '';
    inferPost();
});
function inferPost(){
    setForDetect('api/infer').then(d => {
            //*********************** */
            console.log(d)
            for (bbox of d.bboxes){
                bbox_coord = bbox[0].map(x=> x*ratio);
                drawCtx.fillText(d.labels[bbox[1]]+`(${bbox[2].toFixed(3)})`, bbox_coord[0]+3, bbox_coord[1]-10);
                drawCtx.strokeRect(...bbox_coord);
            }
            //********************** */            
            // document.querySelector('#pred').innerText = d.prediction.join(' ');
            document.querySelector('#plot').src = 'plot.jpg';
            

        }) // 텍스트 출력
        .catch(err => console.error(err));
}

// detect button event
let detectBtn = document.querySelector('#detect');
let isDetecting = false;
detectBtn.addEventListener("click", ()=>{
    if(isDetecting){
        document.querySelector('#pred').innerText = '';
        detectBtn.style.color = 'white';
    }
    else{
        detectPost();        
        document.querySelector('#pred').innerText = 'DETECTING...';
        detectBtn.style.color = 'red';
    }
    isDetecting = !isDetecting;
    console.log('[isDetecting]:', isDetecting);
});
let fpsEl = document.querySelector('#fps');
function detectPost(){
    let startTime = Date.now();
    setForDetect('api/detectweb').then(d => {
        let duration = Date.now()-startTime; // 서버 다녀오는 시간, ms
        let fps = 1/duration*1000;
        fpsEl.innerText = fps.toFixed(2);

        if(radioEl[0].checked){
            //***********box render************ */
            // console.log(d);
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            for (bbox of d.bboxes){
                bbox_coord = bbox[0].map(x=> x = x*ratio);
                drawCtx.fillText(d.labels[bbox[1]]+`(${bbox[2].toFixed(3)})`, bbox_coord[0]+3, bbox_coord[1]-12);
                drawCtx.strokeRect(...bbox_coord);
            }
        }
        // for smoother
        if(onlyOneEl.checked) smoother.renewQueue(d.bboxes.slice(0,1)); // 검출된것 중 오로지 하나만 카드 만듦
        else smoother.renewQueue(d.bboxes); // 검출된것 전체로 카드 만듦
        smoother.showDetectedClass();
        //********************************* */      

        if(isDetecting) detectPost();
        else drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    }) // 텍스트 출력
    .catch(err => console.error(err));
}
// -------------------------  detection apic -----------------------------


// -------------------------  simple face recog opencv -----------------------------
function setForfaceRecog(url){
    return new Promise((resolve, reject) => {
        imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
        imageCanvas.toBlob((file) => {
            let formdata = new FormData();
            formdata.append("image", file);
            formdata.append("mirror", mirror);
            let myInit = {
                method: 'POST',
                body: formdata
            }

            fetch(url, myInit).then((res) => {
                if (res.status === 200 || res.status === 201) { // 성공을 알리는 HTTP 상태 코드면
                    resolve(res.json())
                } else { // 실패를 알리는 HTTP 상태 코드면
                    console.error(res.statusText);
                    reject(res.statusText);
                }
            })
        }, 'image/jpeg');
    });
}
let faceRecogBtn = document.querySelector('#faceRecog');
let isFaceRecognating = false;
faceRecogBtn.addEventListener('click', ()=>{
    if(isFaceRecognating){
        document.querySelector('#pred').innerText = '';
        faceRecogBtn.style.color = 'white';
    }
    else{
        FaceRecogPost();        
        document.querySelector('#pred').innerText = 'Face Recognating...';
        faceRecogBtn.style.color = 'red';
    }
    isFaceRecognating = !isFaceRecognating;
    console.log('[isFaceRecognating]:', isFaceRecognating);
});
function FaceRecogPost(){
    let startTime = Date.now();
    setForfaceRecog('api/faceRecog').then(d => {
        let duration = Date.now()-startTime; // 서버 다녀오는 시간, ms
        let fps = 1/duration*1000;
        fpsEl.innerText = fps.toFixed(2);

        if(radioEl[0].checked){
            //***********box render************ */
            drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            drawCtx.strokeStyle = "rgba(0,0,255,0.5)";
            for (bbox of d.bboxes){
                bbox_coord = bbox.map(x=> x = x*ratio);
                drawCtx.strokeRect(...bbox_coord);
            }
            drawCtx.strokeStyle = "rgba(0,255,0,0.5)";
            //********************************* */
        }      

        //**********얼굴 검출 후 할일************* */
        if(d.face>0){
            console.log('[face]', d); //  검출된 얼굴 개수

            if(window.currTabIdx===1){
                // 얼굴 검출 반복 정지 stop face recog API
                isFaceRecognating = false;
                document.querySelector('#pred').innerText = '';
                faceRecogBtn.style.color = 'white';
                
                // 코멘트
                let commentEl = document.querySelector('#comment')
                commentEl.innerText = `다이애나님 반갑습니다.`;
                Smoother.sleep(2000).then(()=>commentEl.innerText=''); // 3000ms뒤 소멸

                // 화장품 검출 시작
                let clickEvent = new MouseEvent('click', {
                    view: window,
                    bubbles: true,
                    cancelable: true
                });
                detectBtn.dispatchEvent(clickEvent);
                const detectModeEl = document.querySelector('#detectMode');
                detectModeEl.innerHTML = 'T';
            }
        }
        //********************** */
        if(isFaceRecognating) FaceRecogPost();
        else drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    }) // 텍스트 출력
    .catch(err => console.error(err));
}
// -------------------------  simple face recog opencv -----------------------------


/**
 * Make tempoal window for data smoothing
 */
class Smoother{
    constructor(timeWindow){
        this.timeWindow = timeWindow;
        let startTime = Date.now();
        setForDetect('api/detectweb').then(d => {
            this.duration = Date.now()-startTime; // 서버 다녀오는 시간, ms
            this.labels = d.labels.map(x=>x.replace(/\s/g, '_'));
            this.classNum = this.labels.length;
            this.threshold = 0.6;
            this.sequenceNum = Math.round(this.timeWindow/this.duration) || 1; // sequenceNum = 물체 노출시간(timWindow)/duration, ex) 20/10 = 2sec
            this.smoothDataQueue = [];
            this.detctedItemHistory = [];
            for(let i=0; i<this.classNum; i++) this.smoothDataQueue.push(new Array(this.sequenceNum).fill(0))
            this.classStateList = new Array(this.labels.length).fill(false); // 현재 스무딩 데이터가 threshold값을 넘었는지 각 class마다의 상태를 저장해두는 리스트
            this.prevClassStateList = new Array(this.labels.length).fill(false); // 현재 스무딩 데이터가 threshold값을 넘었는지 각 class마다의 상태를 저장해두는 리스트
            console.log('[Smoother]', '<labels>', this.labels, '<timeWindow>', this.timeWindow, '<seqNum>', this.sequenceNum);
            document.querySelector('#screenInfo').innerText += `\n <timeWindow> ${this.timeWindow}, <seqNum> ${this.sequenceNum}`;
        });
    }

    renewQueue = (bboxes) => {
        // delete all first queue item
        for (let q of this.smoothDataQueue){
            q.shift();
            q.push(0);
        }
        for (let bbox of bboxes){
            let detctedQueue = this.smoothDataQueue[bbox[1]];
            detctedQueue[this.sequenceNum-1] = 1;
        }
        this._setclassStateList();
        return
    }

    _setclassStateList = () => {
        this.prevClassStateList = this.classStateList.slice(); // deep copy
        this.smoothDataQueue.map((q, idx) => {
            let mean = q.reduce((a,b)=>a+b)/this.sequenceNum;
            if(mean >= this.threshold) this.classStateList[idx] = true;
            else this.classStateList[idx] = false;
        });
        return;
    }  

    // 검출된 아이템 카드 렌더링
    showDetectedClass = () => {
        let _currDetectedList = [];
        this.classStateList.map((state, idx)=>{
            if(state != this.prevClassStateList[idx]){
                console.log(`${this.labels[idx]} is ${state}...`);
                if(state){
                    _currDetectedList.push(this.labels[idx]);
                    if(window.tabInstance.index===1){ // for scenario-page
                        let commentEl = document.querySelector('#comment')
                        commentEl.innerText = `${this.labels[idx]}를 사용하시는 군요.`
                        Smoother.sleep(2000).then(()=>commentEl.innerText=''); // 3000ms뒤 소멸
                        M.toast({
                            html:`${this.labels[idx]}는 울릉도 해양 심층수의 수분 밸런스와 천연 효소 각질 제거로 더욱 순하게 케어해줘요.`,
                            classes: 'rounded my-toast',
                        });

                        // 새로운 화장품 등록
                        if(window.myItemList.indexOf(this.labels[idx])===-1){
                            var toastHTML = `
                            <span>새로운 화장품이네요. 나만의 화장품에 등록하시겠습니까?</span>
                            <button class="btn-flat toast-action" onclick="window.myItemList.push('${this.labels[idx]}'); M.toast({html:'나만의 화장대에 ${this.labels[idx]} 등록 완료!', classes: 'rounded my-toast'})">OK</button>
                            `;
                            M.toast({
                                html: toastHTML,
                                classes: 'rounded my-toast',
                                displayLength: 6000
                            });
                        }

                        // 프라엘 사용 경고
                        if(this.labels[idx]==="LG-praL"){
                            let _prevDetectedList = this.detctedItemHistory[this.detctedItemHistory.length-1] || [];
                            let _predPlusCurr = _prevDetectedList.concat(_currDetectedList);
                            _predPlusCurr = _predPlusCurr.filter(x => x!="LG-praL"); // 프라엘들은 제거

                            if(_predPlusCurr.length>0){ // 일단은 프라엘 빼고 뭐라도 하나 썼으면 작동
                                commentEl.innerText = `다이애나님 잠시만요.\n사용을 멈춰주세요`
                                Smoother.sleep(2000).then(()=>commentEl.innerText=''); // 3000ms뒤 소멸
                                var toastHTML = `
                                <span>프라엘 관련 자세한 내용 보기</span>
                                <button data-target="pral-modal" class="btn-flat toast-action modal-trigger"">OK</button>
                                `;
                                M.toast({
                                    html: toastHTML,
                                    classes: 'rounded my-toast',
                                    displayLength: 6000
                                });
                            }

                        }
                    }
                    else this._makeCard(this.labels[idx]); // for test-page
                }
                else{
                    if(window.tabInstance.index!==1) this._deleteCard(this.labels[idx]); // for test-page
                }
            }
        });
        // add deteted history
        if(_currDetectedList.length>0) this.detctedItemHistory.push(_currDetectedList);
        return;
    }

    _makeCard = (label) => {
        let cntrElement = document.querySelector('#cards');
        let card = `
        <div id="card_${label}" class="col s4">
            <div class="card">
                <div class="card-image">
                    <img src="images_ext/${label}/${label}_1.jpg" style="height:25vw">
                    <span class="card-title" style="top:0; padding-top:1rem; color:white; font-weight:bold; -webkit-text-stroke:1px black;">${label}</span>
                </div>
            </div>
        </div>
        `;
        cntrElement.innerHTML += card;
        return;
    }

    _deleteCard = (label) => {
        let el_id = `card_${label}`;
        let el = document.querySelector(`#${el_id}`);
        el.parentElement.removeChild(el);
        return;
    }

    _makeComment = (label) => {
        
    }
  
    static sleep = (time) => {
      return new Promise(function(resolve){
        setTimeout(resolve, time);
      });
    }
  }


// Start object detection
let ratio;
function startObjectDetection() {

    console.log("starting object detection");

    //Set canvas sizes base don input video
    drawCanvas.width = v.videoWidth;
    drawCanvas.height = v.videoHeight;
    videoWrapper.style.height = v.videoHeight + "px";

    imageCanvas.width = uploadWidth;
    imageCanvas.height = uploadWidth * (v.videoHeight / v.videoWidth);

    //Some styles for the drawcanvas
    drawCtx.lineWidth = 2;
    drawCtx.strokeStyle = "rgba(0,255,0,0.5)";
    drawCtx.font = "15px Verdana";
    drawCtx.fillStyle = "rgba(0,255,0,0.8)";

    //Save and send the first image
    ratio = drawCanvas.width/imageCanvas.width
    document.querySelector('#screenInfo').innerText += `\n width: ${drawCanvas.width}, height: ${drawCanvas.height}`;
    Smoother.sleep(1500).then(()=>{
        window.smoother = new Smoother(1200);
    });
}
//Starting events

//check if metadata is ready - we need the video size
v.onloadedmetadata = () => {
    console.log("video metadata ready");
    gotMetadata = true;
    if (isPlaying)
        startObjectDetection();
};

//see if the video has started playing
v.onplaying = () => {
    console.log("video playing");
    isPlaying = true;
    if (gotMetadata) {
        startObjectDetection();
    }
};

// ----------------for video record-----------
const recordButton = document.querySelector('#record');
recordButton.addEventListener('click', () => {
    if (recordButton.attributes.isRecording.value === 'no') {
        terminateAllAPI(); // 기존 디텍팅 정지
        recordButton.attributes.isRecording.value = 'yes';
        recordButton.classList.remove('green');
        recordButton.classList.add('red');
        let _itemName = document.querySelector("#itemName").value;
        if(!_itemName){
            alert("Insert an item name!");
            return;
        }
        startRecording();
    } else {
        stopRecording();
        recordButton.classList.remove('red');
        recordButton.classList.add('green');
        uploadVideo();
        recordButton.attributes.isRecording.value = 'no';
    }
});

// https://stackoverflow.com/questions/44392027/webrtc-convert-webm-to-mp4-with-ffmpeg-js - mp4로 녹화
let recordedBlobs;
function startRecording() {
    recordedBlobs = [];
    let options = { mimeType: 'video/webm;codecs=vp9' };

    try {
        mediaRecorder = new MediaRecorder(window.stream, options);
    } catch (e) {
        alert('Exception while creating MediaRecorder:', e);
        // console.error('Exception while creating MediaRecorder:', e);
        return;
    }

    console.log('Created MediaRecorder', mediaRecorder, 'with options', options);
    mediaRecorder.onstop = (event) => {
        console.log('Recorder stopped: ', event);
    };
    mediaRecorder.ondataavailable = handleDataAvailable;
    mediaRecorder.start(10); // collect 10ms of data
    console.log('MediaRecorder started', mediaRecorder);
}

function handleDataAvailable(event) {
    if (event.data && event.data.size > 0) {
        recordedBlobs.push(event.data);
    }
}

function stopRecording() {
    mediaRecorder.stop();
    console.log('Recorded Blobs: ', recordedBlobs);
}

function uploadVideo(){
    let _itemName = document.querySelector("#itemName").value;
    const blob = new Blob(recordedBlobs, {type: 'video/webm'});
    let formdata = new FormData();
    formdata.append("myVideo", blob, _itemName+'.mp4');
    formdata.append("name", _itemName)
    let myInit = {
        method: 'POST',
        body: formdata
    }

    // 디비에 등록
    document.querySelector('#pred').innerText='EXTRACTING...';
    document.querySelector('#spinner').style.display='block';
    registerItem().then((res) => {
        if (res.status === 200 || res.status === 201) { // 성공을 알리는 HTTP 상태 코드면
            return res.json()
        } else { // 실패를 알리는 HTTP 상태 코드면
            console.error(res.statusText);
        }
    }).then(d => {
        console.log(d)
        return fetch('/api/upload', myInit);
    }).then((res) => {
        if (res.status === 200 || res.status === 201) { // 성공을 알리는 HTTP 상태 코드면
            return res.json()
        } else { // 실패를 알리는 HTTP 상태 코드면
            console.error(res.statusText);
        }
    }).then(d => {
        console.log(d)
        document.querySelector('#pred').innerText='FINISHED!';
        document.querySelector('#spinner').style.display='none';
        smoother = new Smoother(1200);
        window.myItemList.push(_itemName); // 내 화장품으로 등록
    }).catch(err => console.error(err));
}

