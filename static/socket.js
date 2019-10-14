/**
 * 현재 소켓 쓰면 서버에서 외부 프로세스 명령어 안먹음...
 * 멀티 프로세스 코딩해야 함
 */
let socket;
let socketBtn = document.querySelector("#socketDetect");
socketBtn.addEventListener("click", () => {
  if (!isDetecting) {
    socket = io.connect(location.origin);
    // event listening
    socket.on("connect", function() {
      socket.emit("user message", {
        data: "[Socket] Connected"
      });
    });

    socket.on("message", function(message) {
      console.log(message);
    });

    socket.on("predict", function(message) {
      let duration = Date.now() - window.ss; // 서버 다녀오는 시간, ms
      let fps = (1 / duration) * 1000;
      fpsEl.innerText = fps.toFixed(2);
      let d = JSON.parse(message);
      //*********************** */
      // console.log(d);
      for (bbox of d.bboxes) {
        bbox_coord = bbox[0].map(x => (x = x * ratio));
        drawCtx.fillText(
          d.labels[bbox[1]] + `(${bbox[2].toFixed(3)})`,
          bbox_coord[0] + 3,
          bbox_coord[1] - 12
        );
        drawCtx.strokeRect(...bbox_coord);
      }
      // for smoother
      // smoother.renewQueue(d.bboxes);
      smoother.renewQueue(d.bboxes.slice(0, 1)); // 오로지 하나만 검출
      smoother.showDetectedClass();
      //********************** */
      if (isDetecting) detectSocket();
    });

    // detect event
    detectSocket();
    document.querySelector("#pred").value = "DETECTING...";
    socketBtn.style.color = "red";
  } else {
    socket.disconnect();
    socket = undefined;
    console.log("[Socket] Disonnected");
    document.querySelector("#pred").value = "";
    socketBtn.style.color = "white";
  }
  isDetecting = !isDetecting;
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  console.log("[isDetecting]:", isDetecting);
});

function detectSocket() {
  window.ss = Date.now();
  return new Promise((resolve, reject) => {
    imageCtx.drawImage(
      v,
      0,
      0,
      v.videoWidth,
      v.videoHeight,
      0,
      0,
      uploadWidth,
      uploadWidth * (v.videoHeight / v.videoWidth)
    );
    imageCanvas.toBlob(file => {
      socket.emit("detect", {
        image: file,
        threshold: threshold.value,
        imageScale: imageScale.value,
        rpnNumX: rpnNumX.value,
        rpnNumY: rpnNumY.value,
        rpnScaleX: rpnScaleX.value,
        rpnScaleY: rpnScaleX.value,
        mirror: mirror
      });
      drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      resolve();
    }, "image/jpeg");
  });
}
