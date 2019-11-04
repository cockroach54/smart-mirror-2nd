
let threshold = document.querySelector("#threshold");
let threshold_c = document.querySelector("#threshold_c");
let interval_c = document.querySelector('#interval_c');
let imageScale = document.querySelector("#imageScale");
let rpnNumX = document.querySelector('#rpnNumX');
let rpnNumY = document.querySelector('#rpnNumY');
let rpnScaleX = document.querySelector('#rpnScaleX');
let rpnScaleY = document.querySelector('#rpnScaleY');

let radioEl = document.getElementsByName('radio-for-box');
let onlyOneEl = document.querySelector('#onlyOne');
let confusedEl = document.querySelector('#confused');

// ********************* Get camera video **********************
let constraints = {
    audio: false,
    // video: {
    //     width: {min: 640, ideal: 1280, max: 1920},
    //     height: {min: 480, ideal: 720, max: 1080}
    // }
    video: {
        // facingMode: facingMode, // mobile에서 적용 "environment":후면카메라, "user":전면카메라
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

// restart webRTC camera
function startWebRTCCamera(constraints){      
    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            window.stream = stream;
            document.getElementById("myVideo").srcObject = stream;
            console.log("Got local user video");
        })
        .catch(err => {
            console.log('navigator.getUserMedia error: ', err)
        });
}

document.querySelector('#flip-cam').addEventListener('click', ()=>{
    if(constraints.video.facingMode==='environment'){
        constraints.video.facingMode='user';
        document.querySelector('video').style.transform = 'scale(1,1)';
    }
    else{
        constraints.video.facingMode='environment';
        document.querySelector('video').style.transform = 'scale(1,1)';
    }
    // stream 먼저 종료 안하면 비동기 순서 에러남
    window.stream.getTracks().forEach(function(track) {
        track.stop();
        isPlaying = false,
        gotMetadata = false;
        startWebRTCCamera(constraints);
      });

});
// ***************************************************************

// mobile setup
let mirror = true;
let r_ww=0.5; r_hh=0.8; r_xx=0.05; r_yy=0.1; r_xx_inverted=1-r_ww-r_xx; // 동영상 좌우 반전 되돌리기
// let r_ww=0.35; r_hh=0.6; r_xx=0.05; r_yy=0.35; r_xx_inverted=1-r_ww-r_xx; // 동영상 좌우 반전 되돌리기
let reg_phone = /Android|webOS|iPhone|iPad|iPod|BlackBerry|BB|PlayBook|IEMobile|Windows Phone|Kindle|Silk|Opera Mini/i;
if (reg_phone.test(navigator.userAgent)) {
    rpnNumX.value = 1;
    rpnNumY.value = 1;
    rpnScaleX.value = 0.6;
    rpnScaleY.value = 0.45;
    threshold.value = 0.88;
    threshold_c.value = 0.86;
    r_ww=0.7; r_hh=0.45; r_xx=0.15; r_yy=0.5; r_xx_inverted=1-r_ww-r_xx;
    // Take the user to a different screen here.
    // 모바일 후면 카메라 미러효과 없애기
    if(constraints.video.facingMode==="environment"){
        document.querySelector('video').style.transform = 'scale(-1,1)';
    }
    // make camera flip button visible
    document.querySelector('#flip-cam').style.display='inline-block';
}

//Parameters
const s = document.getElementById('objDetect');
const sourceVideo = s.getAttribute("data-source");  //the source video to use
const uploadWidth = s.getAttribute("data-uploadWidth") || 640; //the width of the upload file
//Video element selector
v = document.getElementById(sourceVideo);
let ww, hh, xx, yy;

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


function clearNdraw(){
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    // 화면에 항상 고정시킬 그림 여기에
    if(radioEl[0].checked && currTabIdx===0){
        drawCtx.strokeStyle = "rgba(255,0,0,0.5)";  
        drawCtx.strokeRect(xx,yy,ww,hh);
        drawCtx.strokeStyle = "rgba(0,255,0,0.5)";
    }
}

// -------------------------  detection api -----------------------------
function setForDetect(url){
    return new Promise((resolve, reject) => {
        imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
        imageCanvas.toBlob((file) => {
            let formdata = new FormData();
            formdata.append("image", file);
            formdata.append("threshold", threshold.value);
            formdata.append("threshold_c", threshold_c.value);
            formdata.append("interval_c", interval_c.value);
            formdata.append("confuseFlag", confusedEl.checked);
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
            clearNdraw();
            for (bbox of d.bboxes){
                bbox_coord = bbox[0].map(x=> x*ratio);
                drawCtx.fillText(d.labels[bbox[1]]+`(${bbox[2].toFixed(3)})`, bbox_coord[0]+3, bbox_coord[1]-10);
                drawCtx.strokeRect(...bbox_coord);
            }
            //********************** */            
            // document.querySelector('#pred').innerText = d.prediction.join(' ');
            document.querySelector('#plot').src = d.plotPath;
            

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

        // 헷갈리는 부분 등록
        if(d.confused){
            // 두개이상 토스트 안만듦
            if(document.querySelectorAll('.confuse-toast').length < 2){
                console.log(d);
                let idx = d.frames[0][0]; // 맨 처음것만 진행
                let b64 = (d.frames[0][1]).replace(/\n/g, '');
                let toast_id = randomString(8, 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ');
                let toastHTML = `
                <span style="margin-right:10px">헷갈리네요... <span style="color:red; font-weight:bold">${d.labels[idx]}</span>가 맞습니까? 추가 정보로 등록할까요?</span>
                <img src="data:image/png;base64,${b64}" style="height:22vw">
                <button class="btn-flat toast-action" style="margin:0; width:10vw" onclick="handelAddConfusedImage('${b64}', ${idx}, '${toast_id}')">OK</button>
                `;
                M.toast({
                    html: toastHTML,
                    classes: `rounded my-toast ${toast_id} confuse-toast`,
                    displayLength: 6000
                });
            }
        }
        if(radioEl[0].checked){
            //***********box render************ */
            // console.log(d);
            clearNdraw();
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
        else clearNdraw();
    }) // 텍스트 출력
    .catch(err => console.error(err));
}

// confused api call
function handelAddConfusedImage(b64, idx, toast_id){
    // toast 삭제
    let toastElement = document.querySelector(`.${toast_id}`);
    let toastInstance = M.Toast.getInstance(toastElement);
    toastInstance.dismiss();
    // 디텍팅 중지
    document.querySelector('#cards').innerHTML=''
    terminateAllAPI();
    document.querySelector('#pred').innerText='EXTRACTING...';
    document.querySelector('#spinner').style.display='block';
    document.querySelector('#barrier').style.display='initial';
    // 500ms 후 등록
    setTimeout(()=>{
        postConfused(b64, idx).then(res => {
            console.log(res);
            M.toast({html:'추가 사진으로 등록 완료!', classes: 'rounded my-toast'});
            document.querySelector('#pred').innerText='FINISHED!';
            document.querySelector('#spinner').style.display='none';
            document.querySelector('#barrier').style.display='none';
            smoother = new Smoother(1200);
        });
    }, 500);
}
function postConfused(b64, idx){
    return new Promise((resolve, reject) => {
        let formdata = new FormData();
        formdata.append("image_b64", b64); // image
        formdata.append("idx", idx); // image class index
        let myInit = {
            method: 'POST',
            body: formdata
        }
        let url='api/confused';

        fetch(url, myInit).then((res) => {
            if (res.status === 200 || res.status === 201) { // 성공을 알리는 HTTP 상태 코드면
                resolve(res.json())
            } else { // 실패를 알리는 HTTP 상태 코드면
                console.error(res.statusText);
                reject(res.statusText);
            }
        });
    });
}
// -------------------------  detection api -----------------------------


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
            clearNdraw();
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
        else clearNdraw();
    }) // 텍스트 출력
    .catch(err => console.error(err));
}
// -------------------------  simple face recog opencv -----------------------------


// Start web rtc camera
let ratio;
function startObjectDetection() {

    console.log("starting object detection");

    //Set canvas sizes base don input video
    drawCanvas.width = v.videoWidth;
    drawCanvas.height = v.videoHeight;
    videoWrapper.style.height = v.videoHeight + "px";
    videoWrapper.style.width = v.videoWidth + "px";

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
        // record rect setting
        ww = v.videoWidth*r_ww;
        hh = v.videoHeight*r_hh;
        xx = v.videoWidth*r_xx;
        yy = v.videoHeight*r_yy;
        clearNdraw();
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
    formdata.append("name", _itemName);
    formdata.append("cropScale", [r_xx_inverted, r_yy, r_ww, r_hh]);
    let myInit = {
        method: 'POST',
        body: formdata
    }

    // 디비에 등록
    document.querySelector('#pred').innerText='EXTRACTING...';
    document.querySelector('#spinner').style.display='block';
    document.querySelector('#barrier').style.display='initial';
    fetch('/api/upload', myInit).then((res) => {
        if (res.status === 200 || res.status === 201) { // 성공을 알리는 HTTP 상태 코드면
            return res.json()
        } else { // 실패를 알리는 HTTP 상태 코드면
            console.error(res.statusText);
        }
    }).then(d => {
        console.log(d)
        document.querySelector('#pred').innerText='FINISHED!';
        document.querySelector('#spinner').style.display='none';
        document.querySelector('#barrier').style.display='none';
        smoother = new Smoother(1200);
        window.myItemList.push(_itemName); // 내 화장품으로 등록
    }).catch(err => console.error(err));
}

function randomString(length, chars) {
    var result = '';
    for (var i = length; i > 0; --i) result += chars[Math.floor(Math.random() * chars.length)];
    return result;
}