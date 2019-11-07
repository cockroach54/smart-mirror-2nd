/**
 * Make tempoal window for data smoothing
 * Main data state manager!!!
 */
class Smoother{
  constructor(timeWindow){
      this.timeWindow = timeWindow;
      let startTime = Date.now();
      setForDetect('api/detectweb').then(d => {
          this.duration = Date.now()-startTime; // 서버 다녀오는 시간, ms
          this.labels = d.labels.map(x=>x.replace(/\s/g, '__')); // 스페이스 html 아이디 사용불가 보정
          this.classNum = this.labels.length;
          this.threshold = 0.6; // 큐 평균값 threshold
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
      let label_orig = label.replace(/__/g, ' ');
      let card = `
      <div id="card_${label}" class="col s4">
          <div class="card">
              <div class="card-image">
                  <img src="images_ext/${label_orig}/${label_orig}_1.jpg" style="height:25vw">
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
      if(el) el.parentElement.removeChild(el);
      return;
  }

  static sleep = (time) => {
    return new Promise(function(resolve){
      setTimeout(resolve, time);
    });
  }
}
